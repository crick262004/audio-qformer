#!/usr/bin/env python3
"""
Distributed evaluation script for Audio Q-Former checkpoints.
Runs inference on test set across multiple GPUs and computes metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def gather_predictions(local_preds: List[dict], world_size: int) -> List[dict]:
    """Gather predictions from all processes."""
    if world_size == 1:
        return local_preds
    
    # Serialize to JSON strings
    local_json = [json.dumps(p) for p in local_preds]
    
    # Gather all predictions
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_json)
    
    # Flatten and deserialize
    all_preds = []
    for proc_preds in gathered:
        all_preds.extend([json.loads(p) for p in proc_preds])
    
    return all_preds

from train_qformer_llm import (
    AlignHeads,
    JsonlSummDataset,
    Sample,
    collate_to_fbank,
    _build_eval_batch,
)
from audio_llm_bridge import AudioQFormer, AudioQFormerConfig, SimpleProjector

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. WER computation will be skipped.")
    print("Install with: pip install jiwer")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. ROUGE computation will be skipped.")
    print("Install with: pip install rouge-score")


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics including WER, CER, and ROUGE."""
    metrics = {}
    
    # Basic counts
    metrics["num_samples"] = len(predictions)
    
    # Filter valid pairs
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    metrics["num_valid_samples"] = len(valid_pairs)
    
    if not valid_pairs:
        return metrics
    
    preds, refs = zip(*valid_pairs)
    preds, refs = list(preds), list(refs)
    
    # Average prediction/reference length
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    metrics["avg_pred_words"] = sum(pred_lens) / len(pred_lens)
    metrics["avg_ref_words"] = sum(ref_lens) / len(ref_lens)
    metrics["length_ratio"] = metrics["avg_pred_words"] / max(1, metrics["avg_ref_words"])
    
    # WER/CER if available
    if JIWER_AVAILABLE:
        try:
            metrics["wer"] = jiwer.wer(refs, preds)
            metrics["cer"] = jiwer.cer(refs, preds)
        except Exception as e:
            print(f"Warning: WER/CER computation failed: {e}")
    
    # ROUGE scores if available (important for summarization!)
    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(preds, refs):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            metrics["rouge1"] = sum(rouge1_scores) / len(rouge1_scores)
            metrics["rouge2"] = sum(rouge2_scores) / len(rouge2_scores)
            metrics["rougeL"] = sum(rougeL_scores) / len(rougeL_scores)
        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}")
    
    # Exact match (useful for ASR-like tasks)
    exact_matches = sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower())
    metrics["exact_match_rate"] = exact_matches / len(preds)
    
    return metrics


def load_models(
    checkpoint_dir: Path,
    model_name: str = "seamlessM4T_v2_large",
    llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    qformer_layers: int = 2,
    qformer_heads: int = 8,
    qformer_queries: int = 16,
    contrastive_dim: int = 768,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple:
    """Load all models and checkpoint weights."""
    
    from seamless_communication.inference.translator import Translator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if is_main_process():
        print(f"Loading SeamlessM4T model: {model_name}")
    translator = Translator(
        model_name_or_card=model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )
    translator.model.eval()
    enc_dim = translator.model.model_dim
    
    if is_main_process():
        print(f"Loading LLaMA model: {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For distributed, load to specific device (not device_map="auto")
    llama = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    llama.eval()
    
    emb_layer = llama.get_input_embeddings()
    llm_dim = emb_layer.weight.size(1)
    
    if is_main_process():
        print("Creating Q-Former and Projector")
    qcfg = AudioQFormerConfig(
        model_dim=enc_dim,
        num_layers=qformer_layers,
        num_heads=qformer_heads,
        num_queries=qformer_queries,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    qformer = AudioQFormer(qcfg).to(device=device, dtype=dtype)
    
    projector = SimpleProjector(enc_dim, llm_dim).to(device=device, dtype=dtype)
    
    align = AlignHeads(
        d_q=enc_dim,
        d_t=llm_dim,
        d_con=contrastive_dim,
    ).to(device=device, dtype=dtype)
    
    # Load checkpoint weights
    if is_main_process():
        print(f"Loading checkpoint from: {checkpoint_dir}")
    qformer.load_state_dict(torch.load(checkpoint_dir / "qformer.pt", map_location=device))
    projector.load_state_dict(torch.load(checkpoint_dir / "projector.pt", map_location=device))
    if (checkpoint_dir / "align_heads.pt").exists():
        align.load_state_dict(torch.load(checkpoint_dir / "align_heads.pt", map_location=device))
    
    qformer.eval()
    projector.eval()
    align.eval()
    
    return translator, tokenizer, llama, emb_layer, qformer, projector, align


@torch.no_grad()
def run_evaluation(
    checkpoint_dir: Path,
    test_jsonl: Path,
    output_dir: Path,
    batch_size: int = 4,
    gen_max_new_tokens: int = 256,
    max_samples: int = 0,  # 0 = all samples
    model_name: str = "seamlessM4T_v2_large",
    llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    qformer_layers: int = 2,
    qformer_heads: int = 8,
    qformer_queries: int = 16,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """Run distributed evaluation on test set."""
    
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float16
    
    if is_main_process():
        print(f"[Rank {rank}] Loading models...")
    
    # Load models
    translator, tokenizer, llama, emb_layer, qformer, projector, align = load_models(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        llm_model_name=llm_model_name,
        qformer_layers=qformer_layers,
        qformer_heads=qformer_heads,
        qformer_queries=qformer_queries,
        device=device,
        dtype=dtype,
    )
    
    # Prepare separator embedding (must match training: "<AUD>" token)
    sep_ids = tokenizer("<AUD>", add_special_tokens=False, return_tensors="pt").input_ids.to(emb_layer.weight.device)
    with torch.no_grad():
        sep_emb_static = emb_layer(sep_ids)
    
    # Load dataset
    if is_main_process():
        print(f"Loading test dataset: {test_jsonl}")
    ds = JsonlSummDataset(test_jsonl)
    
    if max_samples > 0:
        ds.samples = ds.samples[:max_samples]
        if is_main_process():
            print(f"Limited to {max_samples} samples")
    
    def collate_fn(samples: List[Sample]) -> List[Sample]:
        return samples
    
    # Use DistributedSampler to split data across GPUs
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=collate_fn)
    
    local_predictions = []
    samples_per_gpu = len(ds) // world_size + (1 if len(ds) % world_size > rank else 0)
    
    if is_main_process():
        print(f"\n[DISTRIBUTED] Running on {world_size} GPUs")
        print(f"[DISTRIBUTED] Total samples: {len(ds)}, ~{samples_per_gpu} per GPU")
        print(f"[DISTRIBUTED] Estimated speedup: ~{world_size}x faster\n")
    
    # Sync before starting
    if world_size > 1:
        dist.barrier()
    
    desc = f"GPU {rank}" if world_size > 1 else "Evaluating"
    pbar = tqdm(loader, desc=desc, disable=not is_main_process())
    
    for batch in pbar:
        # Feature extraction
        seqs, padding_mask, targets, insts, audio_paths = collate_to_fbank(translator, batch)
        seqs = seqs.to(device=device, dtype=dtype)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
        
        # Encode audio
        enc_out, enc_pad = translator.model.encode_speech(seqs, padding_mask)
        enc_out = enc_out.to(dtype=dtype)
        
        # Q-Former + Projector
        audio_tokens = qformer(enc_out, None)
        audio_embeds = projector(audio_tokens).to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)
        
        # Build generation inputs
        gen_inputs, _, gen_attn, _ = _build_eval_batch(
            tokenizer,
            emb_layer,
            sep_emb_static,
            audio_embeds,
            insts,
            None,
            None,
            emb_layer.weight.device,
        )
        
        # Generate
        with torch.cuda.amp.autocast(enabled=True):
            gen_out = llama.generate(
                inputs_embeds=gen_inputs,
                attention_mask=gen_attn,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode predictions - ONLY the newly generated tokens (exclude input embeddings)
        input_len = gen_inputs.size(1)
        gen_tokens_only = gen_out[:, input_len:]
        preds = tokenizer.batch_decode(gen_tokens_only, skip_special_tokens=True)
        
        # Store predictions locally
        for audio_path, pred, ref in zip(audio_paths, preds, targets):
            local_predictions.append({
                "audio": audio_path,
                "prediction": pred.strip(),
                "reference": ref,
            })
    
    # Sync and gather all predictions
    if world_size > 1:
        dist.barrier()
        if is_main_process():
            print(f"\n[Rank {rank}] Gathering predictions from all GPUs...")
        all_predictions_raw = gather_predictions(local_predictions, world_size)
    else:
        all_predictions_raw = local_predictions
    
    # Only main process computes metrics and saves
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove duplicates (distributed sampler might pad)
        seen = set()
        all_predictions_deduped = []
        for p in all_predictions_raw:
            key = p["audio"]
            if key not in seen:
                seen.add(key)
                all_predictions_deduped.append(p)
        
        # Save predictions
        predictions_path = output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for rec in all_predictions_deduped:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Extract for metrics
        all_preds = [p["prediction"] for p in all_predictions_deduped]
        all_refs = [p["reference"] for p in all_predictions_deduped]
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(all_preds, all_refs)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*60)
        print(f"\nPredictions saved to: {predictions_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        # Show examples
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        for i in range(min(5, len(all_preds))):
            print(f"\n--- Sample {i+1} ---")
            print(f"Reference: {all_refs[i][:200]}...")
            print(f"Prediction: {all_preds[i][:200]}...")
        
        return metrics
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Distributed evaluation for Audio Q-Former")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to checkpoint directory containing qformer.pt, projector.pt"
    )
    parser.add_argument(
        "--test_jsonl", 
        type=str, 
        default="test.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="eval_output",
        help="Output directory for predictions and metrics"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to evaluate (0=all)")
    parser.add_argument("--model_name", type=str, default="seamlessM4T_v2_large")
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--qformer_layers", type=int, default=2)
    parser.add_argument("--qformer_heads", type=int, default=8)
    parser.add_argument("--qformer_queries", type=int, default=16)
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    if rank == 0:
        print("="*60)
        print("DISTRIBUTED AUDIO Q-FORMER EVALUATION")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Distributed: {is_distributed}")
        print("="*60)
    
    try:
        run_evaluation(
            checkpoint_dir=Path(args.checkpoint_dir),
            test_jsonl=Path(args.test_jsonl),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            gen_max_new_tokens=args.gen_max_new_tokens,
            max_samples=args.max_samples,
            model_name=args.model_name,
            llm_model_name=args.llm_model_name,
            qformer_layers=args.qformer_layers,
            qformer_heads=args.qformer_heads,
            qformer_queries=args.qformer_queries,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

