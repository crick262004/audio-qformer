import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from seamless_communication.inference.translator import Translator
import os
import pathlib
import torch.nn as nn
import torch``
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM

# --- Model and Data-related classes and functions from the project ---

@dataclass
class AudioQFormerConfig:
    model_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_queries: int = 32
    dropout: float = 0.1

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor, kv_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k, key_padding_mask=kv_padding_mask)
        x = queries + self.dropout(attn_out)
        x = x + self.mlp(x)
        return x

class AudioQFormer(nn.Module):
    def __init__(self, config: AudioQFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(torch.randn(config.num_queries, config.model_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=config.model_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(self, encoder_output: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        x = queries
        for blk in self.blocks:
            x = blk(x, encoder_output, encoder_padding_mask)
        return self.final_norm(x)

class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

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

def load_audio_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16_000)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.transpose(0, 1)
    return wav

def collate_to_fbank(
    translator: Translator,
    audio_path: str,
) -> Tuple[torch.Tensor, Optional[Any]]:
    wav = load_audio_16k(audio_path)
    decoded_audio = {"waveform": wav, "sample_rate": 16_000, "format": -1}
    out = translator.convert_to_fbank(decoded_audio)
    collated = translator.collate([out])
    fbank_seqdata = collated["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(fbank_seqdata)
    return seqs, padding_mask

def _build_inference_batch(
    tokenizer,
    emb_layer,
    sep_emb_static: torch.Tensor,
    audio_embeds: torch.Tensor,
    instruction: str,
    emb_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(emb_device)
    prompt_emb = emb_layer(prompt_ids)
    sep_emb = sep_emb_static
    a_emb = audio_embeds

    inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=emb_device)
    return inputs_embeds, attention_mask

@torch.no_grad()
def inference(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(name)s: %(message)s")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. Load Translator for audio preprocessing
    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )
    translator.model.eval()

    # Initialize processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    # 2. Load Q-Former, Projector, and AlignHeads
    model_dim = translator.model.model_dim
    qcfg = AudioQFormerConfig(model_dim=model_dim, num_layers=2, num_heads=8, num_queries=16)
    qformer = AudioQFormer(qcfg).to(device=device, dtype=dtype)
    
    # LLM
    hf_token = (
        (args.hf_token if getattr(args, "hf_token", None) else None)
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    def _from_pretrained_with_optional_token(load_fn, model_id: str, **kwargs):
        if hf_token:
            try:
                return load_fn(model_id, token=hf_token, **kwargs)
            except TypeError as e:
                # Backward compatibility with older Transformers that expect use_auth_token
                if "unexpected keyword argument 'token'" in str(e):
                    return load_fn(model_id, use_auth_token=hf_token, **kwargs)
                raise
        return load_fn(model_id, **kwargs)

    tokenizer = _from_pretrained_with_optional_token(
        AutoTokenizer.from_pretrained, args.llm_model_name, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llama = _from_pretrained_with_optional_token(
        AutoModelForCausalLM.from_pretrained,
        args.llm_model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    llama.eval()

    emb_layer = llama.get_input_embeddings()
    llm_emb_dim = emb_layer.embedding_dim
    projector = SimpleProjector(in_dim=model_dim, out_dim=llm_emb_dim).to(device=device, dtype=dtype)

    qformer.load_state_dict(torch.load(Path(args.model_path) / "qformer.pt"))
    projector.load_state_dict(torch.load(Path(args.model_path) / "projector.pt"))
    
    qformer.eval()
    projector.eval()

    # Create output directory if it doesn't exist
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for audio_path in args.audio_path:
        # Load audio
        audio_tensor = processor(audio=audio_path, sampling_rate=16000, return_tensors="pt").input_features

        # Generate text
        generated_ids = model.generate(input_features=audio_tensor, max_new_tokens=256)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Save generated text to a file
        output_filename = pathlib.Path(audio_path).stem + ".txt"
        output_filepath = pathlib.Path(args.output_dir) / output_filename
        with open(output_filepath, "w") as f:
            f.write(generated_text)

        print(f"Processed {audio_path}. Generated text saved to {output_filepath}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference with Audio-QFormer model")
    p.add_argument("--audio_path", type=str, nargs='+', required=True, help="Path(s) to the audio file(s)")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results")
    p.add_argument("--model_path", type=str, required=True, help="Path to the directory with model checkpoints (qformer.pt, projector.pt).")
    p.add_argument("--model_name", type=str, default="seamlessM4T_v2_large", help="UnitY model name for audio encoding.")
    p.add_argument("--hf_model_name", type=str, default="facebook/seamless-m4t-v2-large", help="Hugging Face model name for AutoProcessor, AutoTokenizer, and AutoModelForCausalLM.")
    p.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model name for the LLM.")
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token for accessing gated/private models. "
            "If omitted, will use HF_TOKEN or HUGGINGFACEHUB_API_TOKEN env vars or CLI login."
        ),
    )
    p.add_argument("--gen_max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    inference(args)