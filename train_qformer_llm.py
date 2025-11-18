#!/usr/bin/env python3

import argparse
import random
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from math import ceil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio
from torch.utils.data import DataLoader, Dataset, DistributedSampler

DATA_DIR = "."

try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from audio_llm_bridge import (
    AudioQFormer,
    AudioQFormerConfig,
    SimpleProjector,
)
from seamless_communication.inference.translator import Translator
from fairseq2.nn.padding import get_seqs_and_padding_mask

logger = logging.getLogger("train_qformer_llm")


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        # Set RANK and LOCAL_RANK for PyTorch's env:// init_method
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logger.info("Running in single-process mode")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def unwrap_model(model):
    """Unwrap model from DDP wrapper if needed."""
    if isinstance(model, DDP):
        return model.module
    return model


class AlignHeads(nn.Module):
    def __init__(self, d_q: int, d_t: int, d_con: int = 768) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_q), nn.Linear(d_q, d_con), nn.GELU(), nn.Linear(d_con, d_con)
        )
        self.t_proj = nn.Sequential(
            nn.LayerNorm(d_t), nn.Linear(d_t, d_con), nn.GELU(), nn.Linear(d_con, d_con)
        )
        self.match_head = nn.Sequential(
            nn.Linear(2 * d_con, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def forward_proj(self, q_pool: torch.Tensor, t_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qz = F.normalize(self.q_proj(q_pool), dim=-1)
        tz = F.normalize(self.t_proj(t_pool), dim=-1)
        return qz, tz


class ContrastiveQueue:
    def __init__(self, d_con: int, K: int = 32768, device: str = "cuda") -> None:
        self.K = K
        self.registered = False
        self.device = device
        self.ptr = 0
        self.feats = torch.zeros(K, d_con, device=device)

    @torch.no_grad()
    def enqueue(self, tz: torch.Tensor) -> None:
        B = tz.shape[0]
        if B == 0:
            return
        if B >= self.K:
            self.feats.copy_(tz[-self.K :])
            self.ptr = 0
            return
        end = self.ptr + B
        if end <= self.K:
            self.feats[self.ptr:end] = tz
        else:
            remain = self.K - self.ptr
            self.feats[self.ptr:] = tz[:remain]
            self.feats[: B - remain] = tz[remain:]
        self.ptr = (self.ptr + B) % self.K


def contrastive_loss(qz: torch.Tensor, tz: torch.Tensor, queue_feats: Optional[torch.Tensor] = None, logit_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    B = qz.size(0)
    if queue_feats is not None and queue_feats.numel() > 0:
        all_t = torch.cat([tz, queue_feats.detach()], dim=0)
    else:
        all_t = tz
    scale = logit_scale if logit_scale is not None else 1.0
    logits = (qz @ all_t.t())
    logits = logits * scale
    target = torch.arange(B, device=qz.device, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    return loss


def matching_loss(align: AlignHeads, qz: torch.Tensor, tz: torch.Tensor, queue_feats: Optional[torch.Tensor] = None, num_queue_negs: int = 0) -> torch.Tensor:
    B = qz.size(0)

    pos_pairs = torch.cat([qz, tz], dim=-1)
    pos_logits = align.match_head(pos_pairs).squeeze(-1)
    pos_labels = torch.ones(B, device=qz.device)

    idx = torch.randperm(B, device=qz.device)
    tz_shuf = tz[idx]
    neg_pairs_ib = torch.cat([qz, tz_shuf], dim=-1)
    neg_logits_ib = align.match_head(neg_pairs_ib).squeeze(-1)
    neg_labels_ib = torch.zeros(B, device=qz.device)

    if queue_feats is not None and num_queue_negs > 0 and queue_feats.numel() > 0:
        K = min(num_queue_negs, queue_feats.size(0))
        rand_idx = torch.randint(0, queue_feats.size(0), (K,), device=qz.device)
        tz_q = queue_feats[rand_idx]
        q_rep = qz.unsqueeze(1).expand(B, K, qz.size(-1))
        tz_rep = tz_q.unsqueeze(0).expand(B, K, tz_q.size(-1))
        neg_pairs_q = torch.cat([q_rep, tz_rep], dim=-1).reshape(B * K, -1)
        neg_logits_q = align.match_head(neg_pairs_q).squeeze(-1)
        neg_labels_q = torch.zeros(B * K, device=qz.device)

        logits = torch.cat([pos_logits, neg_logits_ib, neg_logits_q], dim=0)
        labels = torch.cat([pos_labels, neg_labels_ib, neg_labels_q], dim=0)
    else:
        logits = torch.cat([pos_logits, neg_logits_ib], dim=0)
        labels = torch.cat([pos_labels, neg_labels_ib], dim=0)

    return F.binary_cross_entropy_with_logits(logits, labels)


def contrastive_loss_bidir(
    qz: torch.Tensor,
    tz: torch.Tensor,
    audio_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if text_feats is None or text_feats.numel() == 0:
        all_t = tz
    else:
        all_t = torch.cat([tz, text_feats.detach()], dim=0)
    if audio_feats is None or audio_feats.numel() == 0:
        all_q = qz
    else:
        all_q = torch.cat([qz, audio_feats.detach()], dim=0)

    B = qz.size(0)
    scale = logit_scale if logit_scale is not None else 1.0

    logits_a2t = scale * (qz @ all_t.t())
    target = torch.arange(B, device=qz.device, dtype=torch.long)
    loss_a2t = F.cross_entropy(logits_a2t, target)

    logits_t2a = scale * (tz @ all_q.t())
    loss_t2a = F.cross_entropy(logits_t2a, target)
    return 0.5 * (loss_a2t + loss_t2a)


@dataclass
class Sample:
    audio_path: str
    summary_text: str
    tgt_lang: Optional[str]
    instruction: Optional[str]


class JsonlSummDataset(Dataset[Sample]):
    def __init__(self, jsonl_path: Path) -> None:
        super().__init__()
        self.samples: List[Sample] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                self.samples.append(
                    Sample(
                        audio_path=str(ex["audio"]),
                        summary_text=str(ex["summary"]),
                        tgt_lang=str(ex.get("tgt_lang")) if ex.get("tgt_lang") is not None else None,
                        instruction=str(ex.get("instruction")) if ex.get("instruction") is not None else None,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def load_audio_16k(path: str) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        if PYDUB_AVAILABLE and path.lower().endswith(('.m4a', '.aac', '.mp4')):
            print(f"torchaudio failed to load {path}, falling back to pydub: {e}")
            try:
                audio = None
                for format_hint in ['m4a', 'mp4', 'aac']:
                    try:
                        audio = AudioSegment.from_file(path, format=format_hint)
                        break
                    except Exception:
                        continue
                
                if audio is None:
                    audio = AudioSegment.from_file(path)
                
                audio = audio.set_channels(1).set_frame_rate(16000)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.sample_width == 2:
                    samples = samples / 32768.0
                elif audio.sample_width == 4:
                    samples = samples / 2147483648.0
                elif audio.sample_width == 1:
                    samples = (samples - 128) / 128.0
                wav = torch.from_numpy(samples).unsqueeze(1)
                return wav
            except Exception as pydub_error:
                print(f"Both torchaudio and pydub failed to load {path}")
                print(f"torchaudio error: {e}")
                print(f"pydub error: {pydub_error}")
                print(f"Skipping file {path} - returning dummy audio")
                dummy_wav = torch.zeros((1600, 1))
                return dummy_wav
        else:
            if not PYDUB_AVAILABLE:
                raise RuntimeError(f"Failed to load {path} with torchaudio and pydub is not available. "
                                 f"Install pydub with: pip install pydub") from e
            else:
                raise e
    
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16_000)
    if wav.dim() == 2:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.transpose(0, 1)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(1)
    else:
        wav = wav.reshape(-1, 1)
    return wav


def collate_to_fbank(
    translator: Translator,
    batch: List[Sample],
) -> Tuple[torch.Tensor, Optional[Any], List[str], List[str], List[str]]:
    decoded_list: List[Dict[str, Any]] = []
    targets: List[str] = []
    inst_and_lang: List[str] = []
    audio_paths: List[str] = []

    for s in batch:
        wav = load_audio_16k(s.audio_path)
        decoded_audio = {"waveform": wav, "sample_rate": 16_000, "format": -1}
        out = translator.convert_to_fbank(decoded_audio)
        decoded_list.append(out)
        targets.append(s.summary_text)
        audio_paths.append(s.audio_path)
        inst = s.instruction or "Summarize the following speech succinctly."
        if s.tgt_lang is not None:
            inst = f"{inst}\nTarget language: {s.tgt_lang}."
        inst_and_lang.append(inst)

    collated = translator.collate(decoded_list)
    fbank_seqdata = collated["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(fbank_seqdata)
    return seqs, padding_mask, targets, inst_and_lang, audio_paths


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class TopKCheckpointManager:
    def __init__(self, save_dir: Path, k: int = 10) -> None:
        self.save_dir = save_dir
        self.k = k
        self.entries: List[Dict[str, Any]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.save_dir / "best_checkpoints.json"
        if self.index_path.exists():
            try:
                self.entries = json.loads(self.index_path.read_text())
            except Exception:
                self.entries = []

    def _persist(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(self.entries, key=lambda x: x["metric"]), f, indent=2)

    def _prune_if_needed(self) -> None:
        if len(self.entries) <= self.k:
            return
        self.entries.sort(key=lambda x: x["metric"])
        while len(self.entries) > self.k:
            worst = self.entries.pop(-1)
            try:
                if worst.get("dir") and Path(worst["dir"]).exists():
                    shutil.rmtree(worst["dir"], ignore_errors=True)
            except Exception:
                pass
        self._persist()

    def maybe_save(
        self,
        metric_value: float,
        tag: str,
        state: Dict[str, Any],
    ) -> Optional[Path]:
        if len(self.entries) >= self.k:
            worst = max(self.entries, key=lambda x: x["metric"])
            if metric_value >= worst["metric"]:
                return None

        ckpt_dir = self.save_dir / f"{tag}_loss{metric_value:.4f}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state["qformer"], ckpt_dir / "qformer.pt")
        torch.save(state["projector"], ckpt_dir / "projector.pt")
        torch.save(state["align"], ckpt_dir / "align_heads.pt")
        meta = {"metric": float(metric_value), "tag": tag, "dir": str(ckpt_dir)}
        self.entries.append(meta)
        self._prune_if_needed()
        self._persist()
        return ckpt_dir


def _build_eval_batch(
    tokenizer,
    emb_layer,
    sep_emb_static: torch.Tensor,
    audio_embeds: torch.Tensor,
    instructions: List[str],
    tgt_ids_batch: Optional[torch.Tensor],
    tgt_attn_mask: Optional[torch.Tensor],
    emb_device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[int]]:
    B = audio_embeds.size(0)
    per_sample_inputs: List[torch.Tensor] = []
    per_sample_labels: List[torch.Tensor] = []
    per_sample_lengths: List[int] = []

    for i in range(B):
        prompt_ids = tokenizer(instructions[i], return_tensors="pt").input_ids.to(emb_device)
        prompt_emb = emb_layer(prompt_ids)
        sep_emb = sep_emb_static
        a_emb = audio_embeds[i : i + 1]

        if tgt_ids_batch is not None and tgt_attn_mask is not None:
            tgt_len = int(tgt_attn_mask[i].sum().item())
            tgt_ids = tgt_ids_batch[i : i + 1, :tgt_len]
            tgt_emb = emb_layer(tgt_ids)
            inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb, tgt_emb], dim=1)
        else:
            inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb], dim=1)

        per_sample_inputs.append(inputs_embeds)
        per_sample_lengths.append(int(inputs_embeds.size(1)))

        if tgt_ids_batch is not None and tgt_attn_mask is not None:
            prefix_len = prompt_emb.size(1) + sep_emb.size(1) + a_emb.size(1)
            labels = torch.full(
                (1, inputs_embeds.size(1)), fill_value=-100, dtype=torch.long, device=emb_device
            )
            labels[:, prefix_len:] = tgt_ids
            per_sample_labels.append(labels)

    max_len = max(per_sample_lengths) if per_sample_lengths else 0
    padded_inputs: List[torch.Tensor] = []
    padded_labels: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []

    pad_id = tokenizer.pad_token_id

    for inputs_embeds, cur_len in zip(per_sample_inputs, per_sample_lengths):
        if cur_len < max_len:
            pad_len = max_len - cur_len
            pad_emb = emb_layer(torch.tensor([[pad_id]], device=emb_device)).expand(1, pad_len, -1)
            inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
            attn = torch.cat(
                [
                    torch.ones((1, cur_len), dtype=torch.long, device=emb_device),
                    torch.zeros((1, pad_len), dtype=torch.long, device=emb_device),
                ],
                dim=1,
            )
            if per_sample_labels:
                pad_labels = torch.full((1, pad_len), fill_value=-100, dtype=torch.long, device=emb_device)
                labels = torch.cat([per_sample_labels[len(padded_labels)], pad_labels], dim=1)
                padded_labels.append(labels)
        else:
            attn = torch.ones((1, cur_len), dtype=torch.long, device=emb_device)
            if per_sample_labels:
                padded_labels.append(per_sample_labels[len(padded_labels)])
        padded_inputs.append(inputs_embeds)
        attention_masks.append(attn)

    inputs_embeds = torch.cat(padded_inputs, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    labels = torch.cat(padded_labels, dim=0) if per_sample_labels else None
    return inputs_embeds, labels, attention_mask, per_sample_lengths


@torch.no_grad()
def evaluate_dataset(
    *,
    split_name: str,
    dataset_jsonl: str,
    translator: Translator,
    tokenizer,
    emb_layer,
    sep_emb_static: torch.Tensor,
    qformer: AudioQFormer,
    projector: SimpleProjector,
    llama: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    train_dtype: torch.dtype,
    amp_enabled: bool,
    batch_size: int,
    gen_max_new_tokens: int,
    gen_limit: int,
    gen_seed: int,
    save_predictions_path: Optional[Path] = None,
    num_preview: int = 3,
) -> Tuple[float, List[Dict[str, str]]]:
    ds = JsonlSummDataset(Path(dataset_jsonl))
    def collate_fn(samples: List[Sample]) -> Tuple[torch.Tensor, Any, List[str], List[str], List[str]]:
        return collate_to_fbank(translator, samples)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    total_loss = 0.0
    total_count = 0
    previews: List[Dict[str, str]] = []

    # Prepare sampling for generation
    ds_len = len(ds)
    limit = max(0, min(int(gen_limit), ds_len))
    if limit > 0:
        rng = random.Random(int(gen_seed))
        selected_indices = set(rng.sample(range(ds_len), limit))
        logger.info(f"{split_name}: generation sampling {limit}/{ds_len} (seed={gen_seed}), indices={sorted(list(selected_indices))}")
    else:
        selected_indices = set()
        logger.info(f"{split_name}: generation sampling disabled (gen_limit={gen_limit})")

    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        save_predictions_path.write_text("")

    llama_was_training = any(p.requires_grad and p.grad is not None for p in llama.parameters()) or llama.training
    llama.eval()

    global_offset = 0
    for seqs, padding_mask, targets, insts, audio_paths in tqdm(loader, desc=f"Evaluating {split_name}"):
        seqs = seqs.to(device=device, dtype=dtype)
        if padding_mask is not None and hasattr(padding_mask, "to"):
            padding_mask = padding_mask.to(device)

        enc_out, enc_pad = translator.model.encode_speech(seqs, padding_mask)
        enc_out = enc_out.to(dtype=train_dtype)
        audio_tokens = qformer(enc_out, None)
        audio_embeds = projector(audio_tokens).to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

        tgt_tok = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
        tgt_ids_batch = tgt_tok.input_ids.to(emb_layer.weight.device)
        tgt_attn_mask = tgt_tok.attention_mask.to(emb_layer.weight.device)

        inputs_embeds, labels, attention_mask, _ = _build_eval_batch(
            tokenizer,
            emb_layer,
            sep_emb_static,
            audio_embeds,
            insts,
            tgt_ids_batch,
            tgt_attn_mask,
            emb_layer.weight.device,
        )

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            lm_loss = out.loss
        bs = inputs_embeds.size(0)
        total_loss += float(lm_loss) * bs
        total_count += bs

        # Run generation only for sampled dataset indices
        local_selected = [i for i in range(bs) if (global_offset + i) in selected_indices]
        if local_selected:
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

            index_tensor = torch.tensor(local_selected, device=gen_inputs.device, dtype=torch.long)
            gen_inputs_sel = gen_inputs.index_select(0, index_tensor)
            gen_attn_sel = gen_attn.index_select(0, index_tensor)

            gen_out = llama.generate(
                inputs_embeds=gen_inputs_sel,
                attention_mask=gen_attn_sel,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            gen_tokens_only = gen_out[:, gen_inputs_sel.size(1):]
            preds_sel = tokenizer.batch_decode(gen_tokens_only, skip_special_tokens=True)

            # Add previews from sampled items only
            for idx_in_batch, pred_text in zip(local_selected, preds_sel):
                if len(previews) >= num_preview:
                    break
                previews.append({
                    "audio": audio_paths[idx_in_batch],
                    "pred": pred_text.strip(),
                    "ref": targets[idx_in_batch],
                })

            # Persist predictions if requested
            if save_predictions_path is not None:
                with save_predictions_path.open("a", encoding="utf-8") as f:
                    for idx_in_batch, pred_text in zip(local_selected, preds_sel):
                        rec = {
                            "audio": audio_paths[idx_in_batch],
                            "prediction": pred_text.strip(),
                            "reference": targets[idx_in_batch],
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        global_offset += bs

    avg_loss = total_loss / max(1, total_count)
    logger.info(f"{split_name} avg_lm_loss={avg_loss:.4f}")

    if llama_was_training:
        llama.train()

    return avg_loss, previews


def train(args: argparse.Namespace) -> None:
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()

    # Only main process should log at INFO level
    log_level = logging.INFO if is_main_process() else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s -- %(name)s: %(message)s")

    # Use local_rank to set device for distributed training
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    use_fp16 = (args.use_fp16 or not args.no_fp16) and torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    amp_enabled = dtype == torch.float16 and device.type == "cuda"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if is_main_process():
        logger.info(f"Training on {world_size} processes with device={device}, dtype={dtype}, amp_enabled={amp_enabled}")

    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )
    translator.model.eval()
    set_requires_grad(translator.model, False)

    model_dim = translator.model.model_dim
    qcfg = AudioQFormerConfig(
        model_dim=model_dim,
        num_layers=args.qformer_layers,
        num_heads=args.qformer_heads,
        num_queries=args.qformer_queries,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    train_dtype = torch.float32 if amp_enabled else dtype
    qformer = AudioQFormer(qcfg).to(device=device, dtype=train_dtype)
    projector: SimpleProjector

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _resolve_llm_repo_id(repo_id: str) -> str:
        rid = repo_id.strip()
        lower = rid.lower()
        aliases = {
            "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "meta-llama/llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        }
        return aliases.get(lower, rid)

    llm_repo_id = _resolve_llm_repo_id(args.llm_model_name)

    hf_token = (
        args.hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    def _from_pretrained_with_optional_token(load_fn, model_id: str, **kwargs):
        if hf_token:
            try:
                return load_fn(model_id, token=hf_token, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'token'" in str(e):
                    return load_fn(model_id, use_auth_token=hf_token, **kwargs)
                raise
        return load_fn(model_id, **kwargs)

    tokenizer = _from_pretrained_with_optional_token(
        AutoTokenizer.from_pretrained, llm_repo_id, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM to specific device (not "auto" to avoid multi-GPU splitting in distributed training)
    llama = _from_pretrained_with_optional_token(
        AutoModelForCausalLM.from_pretrained,
        llm_repo_id,
        torch_dtype=dtype,
        device_map={"": device},  # Explicitly load to current device
    )
    if args.llm_lora is not None and args.llm_lora != "":
        from peft import PeftModel

        llama = PeftModel.from_pretrained(llama, args.llm_lora)
    else:
        set_requires_grad(llama, False)
        llama.eval()

    if args.gradient_checkpointing and hasattr(llama, 'gradient_checkpointing_enable'):
        llama.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")

    emb_layer = llama.get_input_embeddings()
    emb_device = emb_layer.weight.device
    llm_dtype = emb_layer.weight.dtype
    sep_ids = tokenizer("<AUD>", add_special_tokens=False, return_tensors="pt").input_ids.to(emb_device)
    with torch.no_grad():
        sep_emb_static = emb_layer(sep_ids)

    llm_emb_dim = emb_layer.embedding_dim
    projector = SimpleProjector(in_dim=model_dim, out_dim=llm_emb_dim).to(device=device, dtype=train_dtype)

    align = AlignHeads(d_q=model_dim, d_t=llm_emb_dim, d_con=args.contrastive_dim).to(device=device, dtype=train_dtype)
    text_queue = ContrastiveQueue(d_con=args.contrastive_dim, K=args.queue_size, device=str(device))
    audio_queue = ContrastiveQueue(d_con=args.contrastive_dim, K=args.queue_size, device=str(device))

    # Wrap models with DDP for distributed training
    if world_size > 1:
        qformer = DDP(qformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        projector = DDP(projector, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        align = DDP(align, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process():
            logger.info("Wrapped models with DistributedDataParallel")

    params = list(qformer.parameters()) + list(projector.parameters()) + list(align.parameters())
    if any(p.requires_grad for p in llama.parameters()):
        params += [p for p in llama.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    ds = JsonlSummDataset(Path(args.dataset_jsonl))
    def collate_fn(samples: List[Sample]) -> Tuple[torch.Tensor, Any, List[str], List[str], List[str]]:
        return collate_to_fbank(translator, samples)

    # Use DistributedSampler for distributed training
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),  # Don't shuffle when using DistributedSampler
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )

    try:
        from transformers import get_cosine_schedule_with_warmup
    except Exception:
        get_cosine_schedule_with_warmup = None
    updates_per_epoch = ceil(len(loader) / max(1, args.grad_accum_steps))
    num_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(2000, num_steps // 10),
            num_training_steps=num_steps,
        )
        if get_cosine_schedule_with_warmup is not None
        else None
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    step = 0
    qformer.train()
    projector.train()
    align.train()
    llama.train() if any(p.requires_grad for p in llama.parameters()) else llama.eval()

    topk_mgr = TopKCheckpointManager(Path(args.output_dir) / "best", k=args.save_top_k)
    best_dev = float("inf")
    no_improve_epochs = 0

    # Track total training time
    training_start_time = time.time()
    if is_main_process():
        logger.info("=" * 80)
        logger.info(f"Starting training for {args.epochs} epochs")
        logger.info("=" * 80)

    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loader = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process())
        # Track per-epoch progress for half-epoch dev eval
        batches_in_epoch = 0
        total_batches_this_epoch = len(loader)
        half_point = max(1, total_batches_this_epoch // 2)
        half_epoch_done = False
        improved_this_epoch = False

        for seqs, padding_mask, targets, insts, _audio_paths in epoch_loader:
            step += 1
            batches_in_epoch += 1
            seqs = seqs.to(device=device, dtype=dtype)
            if padding_mask is not None and hasattr(padding_mask, "to"):
                padding_mask = padding_mask.to(device)
            with torch.no_grad():
                enc_out, enc_pad = translator.model.encode_speech(seqs, padding_mask)

            enc_out = enc_out.to(dtype=train_dtype)
            audio_tokens = qformer(enc_out, None)
            audio_embeds = projector(audio_tokens)

            per_sample_inputs: List[torch.Tensor] = []
            per_sample_labels: List[torch.Tensor] = []
            per_sample_lengths: List[int] = []

            q_pool = audio_tokens.mean(dim=1)

            with torch.no_grad():
                tgt_tok = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
                tgt_ids_batch = tgt_tok.input_ids.to(emb_device)
                tgt_attn_mask = tgt_tok.attention_mask.to(emb_device)
                t_input_embeds = emb_layer(tgt_ids_batch)
                mask = tgt_attn_mask.float().unsqueeze(-1)
                t_pool = (t_input_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            qz, tz = align.forward_proj(q_pool, t_pool.to(device=device, dtype=train_dtype))
            L_con = contrastive_loss_bidir(
                qz,
                tz,
                audio_feats=audio_queue.feats.to(qz.device),
                text_feats=text_queue.feats.to(qz.device),
                logit_scale=align.logit_scale.clamp(0, 5).exp(),
            )
            L_match = matching_loss(align, qz, tz, queue_feats=text_queue.feats.to(qz.device), num_queue_negs=args.num_queue_negs)

            for i in range(len(targets)):
                instruction = insts[i]
                prompt_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(emb_device)
                prompt_emb = emb_layer(prompt_ids)

                sep_emb = sep_emb_static

                a_emb = audio_embeds[i : i + 1].to(device=emb_device, dtype=llm_dtype)

                tgt_len = int(tgt_attn_mask[i].sum().item())
                tgt_ids = tgt_ids_batch[i : i + 1, :tgt_len]
                tgt_emb = emb_layer(tgt_ids)

                inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb, tgt_emb], dim=1)
                per_sample_inputs.append(inputs_embeds)
                per_sample_lengths.append(int(inputs_embeds.size(1)))

                prefix_len = prompt_emb.size(1) + sep_emb.size(1) + a_emb.size(1)
                labels = torch.full((1, inputs_embeds.size(1)), fill_value=-100, dtype=torch.long, device=emb_device)
                labels[:, prefix_len:] = tgt_ids
                per_sample_labels.append(labels)
            max_len = max(per_sample_lengths) if per_sample_lengths else 0
            padded_inputs: List[torch.Tensor] = []
            padded_labels: List[torch.Tensor] = []
            attention_masks: List[torch.Tensor] = []

            for inputs_embeds, labels, cur_len in zip(per_sample_inputs, per_sample_labels, per_sample_lengths):
                if cur_len < max_len:
                    pad_len = max_len - cur_len
                    pad_id = tokenizer.pad_token_id
                    pad_emb = emb_layer(torch.tensor([[pad_id]], device=inputs_embeds.device)).expand(1, pad_len, -1)
                    inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                    pad_labels = torch.full((1, pad_len), fill_value=-100, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([labels, pad_labels], dim=1)
                    attn = torch.cat([
                        torch.ones((1, cur_len), dtype=torch.long, device=inputs_embeds.device),
                        torch.zeros((1, pad_len), dtype=torch.long, device=inputs_embeds.device),
                    ], dim=1)
                else:
                    attn = torch.ones((1, cur_len), dtype=torch.long, device=inputs_embeds.device)
                padded_inputs.append(inputs_embeds)
                padded_labels.append(labels)
                attention_masks.append(attn)

            inputs_embeds = torch.cat(padded_inputs, dim=0)
            labels = torch.cat(padded_labels, dim=0)
            attention_mask = torch.cat(attention_masks, dim=0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                lm_loss = out.loss

            with torch.no_grad():
                audio_queue.enqueue(qz.detach().to(audio_queue.feats.device))
                text_queue.enqueue(tz.detach().to(text_queue.feats.device))

            total_loss = lm_loss + args.lambda_con * L_con + args.lambda_match * L_match
            loss = total_loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if step % (args.grad_accum_steps * 10) == 0:
                    torch.cuda.empty_cache()
                if scheduler is not None:
                    scheduler.step()

            if step % args.log_every == 0 and is_main_process():
                elapsed_time = time.time() - training_start_time
                hours, remainder = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                loss_info = (
                    f"epoch={epoch} step={step} elapsed={hours:02d}:{minutes:02d}:{seconds:02d} "
                    f"loss={(float(loss) * args.grad_accum_steps):.4f} "
                    f"lm={float(lm_loss):.4f} con={float(L_con):.4f} match={float(L_match):.4f}"
                )
                logger.info(loss_info)
                epoch_loader.set_postfix({
                    'elapsed': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                    'loss': f"{(float(loss) * args.grad_accum_steps):.4f}",
                    'lm': f"{float(lm_loss):.4f}",
                    'con': f"{float(L_con):.4f}",
                    'match': f"{float(L_match):.4f}"
                })

            # Step-based dev eval (only if > 0)
            if (args.dev_eval_steps > 0 and step > 0 and step % args.dev_eval_steps == 0 and 
                args.dev_jsonl is not None and os.path.isfile(args.dev_jsonl)):
                eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
                dev_loss, previews = evaluate_dataset(
                    split_name="dev",
                    dataset_jsonl=args.dev_jsonl,
                    translator=translator,
                    tokenizer=tokenizer,
                    emb_layer=emb_layer,
                    sep_emb_static=sep_emb_static,
                    qformer=qformer,
                    projector=projector,
                    llama=llama,
                    device=device,
                    dtype=dtype,
                    train_dtype=train_dtype,
                    amp_enabled=amp_enabled,
                    batch_size=eval_bs,
                    gen_max_new_tokens=args.gen_max_new_tokens,
                    gen_limit=args.gen_limit_dev,
                    gen_seed=args.gen_seed,
                    save_predictions_path=None,
                    num_preview=args.eval_preview_samples,
                )
                if dev_loss < best_dev:
                    improved_this_epoch = True
                    best_dev = dev_loss
                if is_main_process():
                    for j, ex in enumerate(previews):
                        logger.info(f"[dev sample {j+1}] ref={ex['ref'][:200]} | pred={ex['pred'][:200]} | audio={ex['audio']}")
                    # Unwrap models from DDP before saving
                    state = {
                        "qformer": unwrap_model(qformer).state_dict(),
                        "projector": unwrap_model(projector).state_dict(),
                        "align": unwrap_model(align).state_dict(),
                    }
                    tag = f"epoch{epoch}_step{step}"
                    saved_dir = topk_mgr.maybe_save(dev_loss, tag, state)
                    if saved_dir is not None:
                        logger.info(f"Saved top-k checkpoint to {saved_dir}")
                # Barrier to ensure all processes wait for checkpoint saving
                if dist.is_initialized():
                    dist.barrier()

            # Half-epoch dev eval (default when --dev_eval_steps == -1)
            if (args.dev_eval_steps == -1 and not half_epoch_done and
                args.dev_jsonl is not None and os.path.isfile(args.dev_jsonl) and
                batches_in_epoch >= half_point):
                eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
                dev_loss, previews = evaluate_dataset(
                    split_name="dev",
                    dataset_jsonl=args.dev_jsonl,
                    translator=translator,
                    tokenizer=tokenizer,
                    emb_layer=emb_layer,
                    sep_emb_static=sep_emb_static,
                    qformer=qformer,
                    projector=projector,
                    llama=llama,
                    device=device,
                    dtype=dtype,
                    train_dtype=train_dtype,
                    amp_enabled=amp_enabled,
                    batch_size=eval_bs,
                    gen_max_new_tokens=args.gen_max_new_tokens,
                    gen_limit=args.gen_limit_dev,
                    gen_seed=args.gen_seed,
                    save_predictions_path=None,
                    num_preview=args.eval_preview_samples,
                )
                if dev_loss < best_dev:
                    improved_this_epoch = True
                    best_dev = dev_loss
                if is_main_process():
                    for j, ex in enumerate(previews):
                        logger.info(f"[dev sample {j+1}] ref={ex['ref'][:200]} | pred={ex['pred'][:200]} | audio={ex['audio']}")
                    # Unwrap models from DDP before saving
                    state = {
                        "qformer": unwrap_model(qformer).state_dict(),
                        "projector": unwrap_model(projector).state_dict(),
                        "align": unwrap_model(align).state_dict(),
                    }
                    tag = f"epoch{epoch}_step{step}_half"
                    saved_dir = topk_mgr.maybe_save(dev_loss, tag, state)
                    if saved_dir is not None:
                        logger.info(f"Saved top-k checkpoint to {saved_dir}")
                # Barrier to ensure all processes wait for checkpoint saving
                if dist.is_initialized():
                    dist.barrier()
                half_epoch_done = True

            if args.max_steps > 0 and step >= args.max_steps:
                break

        if args.max_steps > 0 and step >= args.max_steps:
            break

        if (args.dev_eval_steps <= 0 and 
            args.dev_jsonl is not None and os.path.isfile(args.dev_jsonl)):
            eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
            dev_loss, previews = evaluate_dataset(
                split_name="dev",
                dataset_jsonl=args.dev_jsonl,
                translator=translator,
                tokenizer=tokenizer,
                emb_layer=emb_layer,
                sep_emb_static=sep_emb_static,
                qformer=qformer,
                projector=projector,
                llama=llama,
                device=device,
                dtype=dtype,
                train_dtype=train_dtype,
                amp_enabled=amp_enabled,
                batch_size=eval_bs,
                gen_max_new_tokens=args.gen_max_new_tokens,
                gen_limit=args.gen_limit_dev,
                gen_seed=args.gen_seed,
                save_predictions_path=None,
                num_preview=args.eval_preview_samples,
            )
            if dev_loss < best_dev:
                improved_this_epoch = True
                best_dev = dev_loss
            if is_main_process():
                for j, ex in enumerate(previews):
                    logger.info(f"[dev sample {j+1}] ref={ex['ref'][:200]} | pred={ex['pred'][:200]} | audio={ex['audio']}")
                # Unwrap models from DDP before saving
                state = {
                    "qformer": unwrap_model(qformer).state_dict(),
                    "projector": unwrap_model(projector).state_dict(),
                    "align": unwrap_model(align).state_dict(),
                }
                tag = f"epoch{epoch}_step{step}"
                saved_dir = topk_mgr.maybe_save(dev_loss, tag, state)
                if saved_dir is not None:
                    logger.info(f"Saved top-k checkpoint to {saved_dir}")
            # Barrier to ensure all processes wait for checkpoint saving
            if dist.is_initialized():
                dist.barrier()

        # Early stopping check (based on dev improvements within the epoch)
        if args.dev_jsonl is not None and os.path.isfile(args.dev_jsonl):
            if improved_this_epoch:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                logger.info(
                    f"No dev improvement in epoch {epoch+1}. Patience {no_improve_epochs}/{args.early_stop_patience}"
                )
            if no_improve_epochs >= args.early_stop_patience:
                logger.info("Early stopping triggered due to no dev improvement.")
                break

    # Log total training time
    total_training_time = time.time() - training_start_time
    if is_main_process():
        hours, remainder = divmod(int(total_training_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("=" * 80)
        logger.info(f"Training completed! Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logger.info("=" * 80)

    # Save final checkpoints only on main process
    if is_main_process():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Unwrap models from DDP before saving
        torch.save(unwrap_model(qformer).state_dict(), out_dir / "qformer.pt")
        torch.save(unwrap_model(projector).state_dict(), out_dir / "projector.pt")
        torch.save(unwrap_model(align).state_dict(), out_dir / "align_heads.pt")
        logger.info(f"Saved Q-Former, projector, and align heads to {out_dir}")

    # Barrier to ensure all processes wait for final checkpoint saving
    if dist.is_initialized():
        dist.barrier()

    out_dir = Path(args.output_dir)  # Define out_dir for all processes
    if args.test_jsonl is not None and os.path.isfile(args.test_jsonl) and not args.skip_final_test_eval and is_main_process():
        eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
        test_pred_path = out_dir / "test_predictions.jsonl"
        test_loss, previews = evaluate_dataset(
            split_name="test",
            dataset_jsonl=args.test_jsonl,
            translator=translator,
            tokenizer=tokenizer,
            emb_layer=emb_layer,
            sep_emb_static=sep_emb_static,
            qformer=qformer,
            projector=projector,
            llama=llama,
            device=device,
            dtype=dtype,
            train_dtype=train_dtype,
            amp_enabled=amp_enabled,
            batch_size=eval_bs,
            gen_max_new_tokens=args.gen_max_new_tokens,
            gen_limit=args.gen_limit_test,
            gen_seed=args.gen_seed,
            save_predictions_path=test_pred_path,
            num_preview=args.eval_preview_samples,
        )
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"avg_lm_loss": test_loss}, f, indent=2)
        logger.info(f"Wrote test predictions to {test_pred_path} and metrics to test_metrics.json")
    else:
        if is_main_process():
            logger.info("Skipping final test evaluation. Use --skip_final_test_eval to disable or provide --test_jsonl.")

    # Clean up distributed training
    cleanup_distributed()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Q-Former + projector for audio->LLM summarization")
    p.add_argument("--dataset_jsonl", type=str, default=f"{DATA_DIR}/train.jsonl", help="Path to JSONL with {audio, summary, [tgt_lang], [instruction]}")
    p.add_argument("--model_name", type=str, default="seamlessM4T_v2_large", help="UnitY model name")
    p.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model name for Llama")
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token for accessing gated/private models. "
            "If omitted, will use HF_TOKEN or HUGGINGFACEHUB_API_TOKEN env vars or CLI login."
        ),
    )
    p.add_argument("--llm_lora", type=str, default=None, help="Optional LoRA adapter path for Llama")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps (0=disable)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--use_fp16", action="store_true", help="Use mixed precision training")
    p.add_argument("--no_fp16", action="store_true", help="Disable mixed precision (fp16 enabled by default)")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    p.add_argument("--qformer_layers", type=int, default=2)
    p.add_argument("--qformer_heads", type=int, default=8)
    p.add_argument("--qformer_queries", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./qformer_run1")
    p.add_argument("--dev_jsonl", type=str, default=f"{DATA_DIR}/dev.jsonl")
    p.add_argument("--test_jsonl", type=str, default=f"{DATA_DIR}/test.jsonl")
    p.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size (defaults to train batch size)")
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gen_limit_dev", type=int, default=3, help="Number of dev examples to run generation on (random; 0=disable)")
    p.add_argument("--gen_limit_test", type=int, default=10, help="Number of test examples to run generation on (random; 0=disable)")
    p.add_argument("--gen_seed", type=int, default=42, help="Seed for generation sampling")
    p.add_argument("--eval_preview_samples", type=int, default=3)
    p.add_argument("--save_top_k", type=int, default=10)
    p.add_argument("--lambda_con", type=float, default=0.2, help="Weight for contrastive loss")
    p.add_argument("--lambda_match", type=float, default=0.1, help="Weight for matching loss")
    p.add_argument("--contrastive_dim", type=int, default=768, help="Projection dim for contrastive space")
    p.add_argument("--queue_size", type=int, default=32768, help="Size of contrastive feature queue")
    p.add_argument("--num_queue_negs", type=int, default=64, help="Number of queue negatives for matching loss")
    p.add_argument(
        "--dev_eval_steps",
        type=int,
        default=-1,
        help="Dev eval frequency: -1=half-epoch (default), 0=per-epoch, >0=every N steps",
    )
    p.add_argument("--early_stop_patience", type=int, default=2, help="Early stopping patience in epochs (based on dev improvements)")
    p.add_argument("--skip_final_test_eval", action="store_true", help="Skip final test evaluation after training (runs by default if --test_jsonl provided)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
