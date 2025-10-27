#!/usr/bin/env python3
"""
Main training script for the Audio Q-Former model.

This script orchestrates the entire fine-tuning process for bridging a pre-trained
audio encoder (SeamlessM4T) with a Large Language Model (e.g., Llama 3.1). It leverages
a multi-part loss function to learn a meaningful alignment between audio and text
representations.

The core training components are:
1.  A frozen SeamlessM4T model to extract rich audio features.
2.  A trainable AudioQFormer and Projector that learn to distill and map these
    audio features into the LLM's embedding space.
3.  A frozen (or LoRA-finetuned) LLM that generates text summaries based on the
    processed audio context.
4.  A sophisticated loss function combining:
    - Language Modeling Loss (Cross-Entropy) for fluent text generation.
    - Bidirectional Contrastive Loss (InfoNCE) for audio-text alignment.
    - Binary Matching Loss for fine-grained pair discrimination.

The script handles data loading, model setup, the main training loop, periodic
evaluation, and checkpoint management."""

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from math import ceil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "."

# Optional dependency for handling more audio formats like M4A.
try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Import core model architecture components.
from audio_llm_bridge import (
    AudioQFormer,
    AudioQFormerConfig,
    SimpleProjector,
)
# Import the high-level interface for SeamlessM4T.
from seamless_communication.inference.translator import Translator
from fairseq2.nn.padding import get_seqs_and_padding_mask # for handling batched sequences

logger = logging.getLogger("train_qformer_llm")


class AlignHeads(nn.Module):
    """
    Contains the projection and classification heads for the alignment losses.

    This module is central to the contrastive learning framework. It learns to project
    audio and text features into a shared embedding space where their similarity can
    be measured for the InfoNCE loss. It also contains a classifier head to
    predict whether an audio-text pair is a positive match for the matching loss.
    """
    def __init__(self, d_q: int, d_t: int, d_con: int = 768) -> None:
        super().__init__()
        # Projection head for audio (query) features for InfoNCE loss.
        # A small neural network (a projection head) that takes the pooled audio features from the Q-Former and projects them into the shared contrastive space.
        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_q), nn.Linear(d_q, d_con), nn.GELU(), nn.Linear(d_con, d_con)
        )
        # Projection head for text features for InfoNCE loss. Similar to above, but for text features.
        self.t_proj = nn.Sequential(
            nn.LayerNorm(d_t), nn.Linear(d_t, d_con), nn.GELU(), nn.Linear(d_con, d_con)
        )
        # Binary classifier head for the audio-text matching loss.
        # A classifier that takes a concatenated audio-text pair and predicts if they match (a binary classification task for the matching_loss).
        self.match_head = nn.Sequential(
            nn.Linear(2 * d_con, 256), nn.GELU(), nn.Linear(256, 1)
        )
        # A learnable temperature parameter used in the contrastive loss to control the sharpness of the probability distribution, helping to stabilize training.
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def forward_proj(self, q_pool: torch.Tensor, t_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects audio and text features into the shared space and normalizes them.

        This method takes the pooled audio (q_pool) and text (t_pool) features, passes them through their respective projection heads (q_proj, t_proj), and normalizes them. Normalization ensures that the similarity calculation (dot product) is equivalent to cosine similarity.
        """
        qz = F.normalize(self.q_proj(q_pool), dim=-1)
        tz = F.normalize(self.t_proj(t_pool), dim=-1)
        return qz, tz


class ContrastiveQueue:
    """
    A memory bank for storing a large number of feature vectors (negatives).

    This queue provides a richer set of negative samples for contrastive learning
    than what is available in a single batch. It operates as a fixed-size FIFO
    (First-In, First-Out) buffer.
    """
    def __init__(self, d_con: int, K: int = 32768, device: str = "cuda") -> None:
        self.K = K
        self.registered = False
        self.device = device
        self.ptr = 0 # keeps track of the current position to insert new features.
        # Initialize a buffer to store the feature vectors.
        self.feats = torch.zeros(K, d_con, device=device)

    @torch.no_grad()
    def enqueue(self, tz: torch.Tensor) -> None:
        """This method adds new features to the queue in a First-In, First-Out (FIFO) manner. It overwrites the oldest features once the queue is full. The @torch.no_grad() decorator ensures that no gradients are computed for this operation."""
        B = tz.shape[0]
        if B == 0:
            return
        # Handle cases where the batch is larger than the queue.
        if B >= self.K:
            self.feats.copy_(tz[-self.K :])
            self.ptr = 0
            return
        # Find the start and end pointers for insertion.
        end = self.ptr + B
        if end <= self.K:
            # Simple case: fits without wrapping around.
            self.feats[self.ptr:end] = tz
        else:
            # Case: wraps around the end of the buffer.
            remain = self.K - self.ptr
            self.feats[self.ptr:] = tz[:remain]
            self.feats[: B - remain] = tz[remain:]
        # Update the pointer.
        self.ptr = (self.ptr + B) % self.K


# These functions calculate the InfoNCE contrastive loss. The goal is to maximize the similarity between a correct audio-text pair while minimizing its similarity with all other "negative" samples in the batch and in the contrastive queue.
def contrastive_loss(qz: torch.Tensor, tz: torch.Tensor, queue_feats: Optional[torch.Tensor] = None, logit_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates the uni-directional InfoNCE contrastive loss."""
    B = qz.size(0)
    # Combine in-batch negatives with negatives from the queue.
    if queue_feats is not None and queue_feats.numel() > 0:
        all_t = torch.cat([tz, queue_feats.detach()], dim=0)
    else:
        all_t = tz
    scale = logit_scale if logit_scale is not None else 1.0
    # Calculate similarity scores (logits) between audio and all text features.
    logits = (qz @ all_t.t())
    logits = logits * scale
    # The target for each audio sample is its corresponding text sample (at index 0 to B-1).
    target = torch.arange(B, device=qz.device, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    return loss


def matching_loss(align: AlignHeads, qz: torch.Tensor, tz: torch.Tensor, queue_feats: Optional[torch.Tensor] = None, num_queue_negs: int = 0) -> torch.Tensor:
    """
    Calculates the binary audio-text matching (ATM) loss.

    It frames the alignment task as a simple binary classification problem.
    Simple Binary Cross-Entropy Loss (BCE) is used here because the task is a binary classification:

    This loss trains a classifier to distinguish between correct (positive) pairs
    and incorrect (negative) pairs.
    """
    B = qz.size(0)

    # Positive Pairs: It concatenates the correct audio (qz) and text (tz) features and passes them to the align.match_head to get a "match" score. The target label is 1 (True).
    pos_pairs = torch.cat([qz, tz], dim=-1)
    pos_logits = align.match_head(pos_pairs).squeeze(-1)
    pos_labels = torch.ones(B, device=qz.device)

    # Negative Pairs: It creates negative examples by shuffling the text features within the batch (in-batch negatives) and by pairing audio features with random text features from the contrastive queue (queue negatives). The target label for these is 0 (False).
    idx = torch.randperm(B, device=qz.device)
    tz_shuf = tz[idx]
    neg_pairs_ib = torch.cat([qz, tz_shuf], dim=-1)
    neg_logits_ib = align.match_head(neg_pairs_ib).squeeze(-1)
    neg_labels_ib = torch.zeros(B, device=qz.device)

    # Negative pairs (from queue): sample random text features from the queue.
    if queue_feats is not None and num_queue_negs > 0 and queue_feats.numel() > 0:
        K = min(num_queue_negs, queue_feats.size(0))
        rand_idx = torch.randint(0, queue_feats.size(0), (K,), device=qz.device)
        tz_q = queue_feats[rand_idx]
        q_rep = qz.unsqueeze(1).expand(B, K, qz.size(-1))
        tz_rep = tz_q.unsqueeze(0).expand(B, K, tz_q.size(-1))
        neg_pairs_q = torch.cat([q_rep, tz_rep], dim=-1).reshape(B * K, -1)
        neg_logits_q = align.match_head(neg_pairs_q).squeeze(-1)
        neg_labels_q = torch.zeros(B * K, device=qz.device)

        # Combine all logits and labels.
        logits = torch.cat([pos_logits, neg_logits_ib, neg_logits_q], dim=0)
        labels = torch.cat([pos_labels, neg_labels_ib, neg_labels_q], dim=0)
    else:
        logits = torch.cat([pos_logits, neg_logits_ib], dim=0)
        labels = torch.cat([pos_labels, neg_labels_ib], dim=0)

    # Calculate binary cross-entropy loss.
    return F.binary_cross_entropy_with_logits(logits, labels)


def contrastive_loss_bidir(
    qz: torch.Tensor,
    tz: torch.Tensor,
    audio_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the full bidirectional InfoNCE contrastive loss.

    This computes the loss in both directions (audio-to-text and text-to-audio)
    and averages them for a more robust alignment signal.
    Is the one primarily used.
    """
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
    target = torch.arange(B, device=qz.device, dtype=torch.long)

    # Audio-to-text loss: It compares each audio query (qz) to all text features (all_t), aiming for the highest score with its true text partner
    logits_a2t = scale * (qz @ all_t.t())
    target = torch.arange(B, device=qz.device, dtype=torch.long)
    loss_a2t = F.cross_entropy(logits_a2t, target)

    # Text-to-audio loss: It does the reverse, comparing each text query (tz) to all audio features (all_q).
    logits_t2a = scale * (tz @ all_q.t())
    loss_t2a = F.cross_entropy(logits_t2a, target)

    # The final loss is the average of these two directional losses, ensuring robust alignment
    return 0.5 * (loss_a2t + loss_t2a)


@dataclass
class Sample:
    """A simple dataclass to hold one training example. (A simple data structure to hold the information for one training example)"""
    audio_path: str
    summary_text: str
    tgt_lang: Optional[str]
    instruction: Optional[str]


class JsonlSummDataset(Dataset[Sample]):
    """
    A standard PyTorch Dataset to load samples from a .jsonl file.
    
    This class reads a .jsonl file (where each line is a JSON object), parses each line into a Sample object, and serves them up for training.
    """
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
    """
    A robust audio loading function that loads an audio file, resamples it to 16kHz,
    converts it to mono, and returns it as a PyTorch tensor. It includes a fallback
    to the `pydub` library for formats not supported by `torchaudio`.
    """
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        # Fallback to pydub for formats like M4A if available.
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
                
                # Resample to 16kHz and convert to mono.
                audio = audio.set_channels(1).set_frame_rate(16000)
                # Normalize samples to float32 in range [-1, 1].
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
    
    # Resample if necessary.
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16_000)
    # Ensure audio is mono and has shape (samples, 1).
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
    """
    The collate function for the DataLoader. It takes a batch of Samples, loads the
    audio waveforms, and uses the SeamlessM4T processor (via the Translator object)
    to convert them into log-mel filterbank (fbank) features, which are the
    input to the audio encoder.
    """
    decoded_list: List[Dict[str, Any]] = []
    targets: List[str] = []
    inst_and_lang: List[str] = []
    audio_paths: List[str] = []

    for s in batch: # It iterates through the samples, loading each audio file using load_audio_16k.
        wav = load_audio_16k(s.audio_path)
        decoded_audio = {"waveform": wav, "sample_rate": 16_000, "format": -1}
        # It uses the translator.convert_to_fbank method from SeamlessM4T to convert the raw audio waveforms into log-mel filterbank features, which are the standard input for audio transformers.
        out = translator.convert_to_fbank(decoded_audio)
        decoded_list.append(out)
        targets.append(s.summary_text)
        audio_paths.append(s.audio_path)
        # Prepare the instruction prompt for the LLM.
        inst = s.instruction or "Summarize the following speech succinctly."
        if s.tgt_lang is not None:
            inst = f"{inst}\nTarget language: {s.tgt_lang}."
        inst_and_lang.append(inst)

    # (Collation) It bundles these features, text targets, and instructions into a single batch of tensors, handling necessary padding for sequences of different lengths.
    collated = translator.collate(decoded_list)
    fbank_seqdata = collated["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(fbank_seqdata)
    return seqs, padding_mask, targets, inst_and_lang, audio_paths


# Utilities for training and checkpointing.

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """
    A helper function to recursively set the requires_grad attribute of a module's parameters.

    A simple helper function to either freeze or unfreeze all parameters of a given PyTorch module. This is used to freeze the weights of the pre-trained SeamlessM4T encoder and (optionally) the LLM, so that only the Q-Former, projector, and alignment heads are trained.
    """
    for p in module.parameters():
        p.requires_grad = flag


class TopKCheckpointManager:
    """
    A utility to manage model checkpoints, saving only the top 'k' best performing
    models based on a validation metric (e.g., loss). This prevents filling up
    disk space with suboptimal checkpoints.
    """
    def __init__(self, save_dir: Path, k: int = 10) -> None:
        self.save_dir = save_dir
        self.k = k
        self.entries: List[Dict[str, Any]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Load existing checkpoint index if it exists.
        self.index_path = self.save_dir / "best_checkpoints.json"
        if self.index_path.exists():
            try:
                self.entries = json.loads(self.index_path.read_text())
            except Exception:
                self.entries = []

    def _persist(self) -> None:
        """Saves the current index of best checkpoints to a JSON file."""
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(self.entries, key=lambda x: x["metric"]), f, indent=2)

    def _prune_if_needed(self) -> None:
        """If more than 'k' checkpoints exist, it removes the worst one."""
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
        """
        Saves a new checkpoint if its metric is better than the worst of the
        current top 'k' checkpoints.
        """
        if len(self.entries) >= self.k:
            worst = max(self.entries, key=lambda x: x["metric"])
            if metric_value >= worst["metric"]: # Assuming lower is better.
                return None

        ckpt_dir = self.save_dir / f"{tag}_loss{metric_value:.4f}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save the state of the trainable modules.
        torch.save(state["qformer"], ckpt_dir / "qformer.pt")
        torch.save(state["projector"], ckpt_dir / "projector.pt")
        torch.save(state["align"], ckpt_dir / "align_heads.pt")
        meta = {"metric": float(metric_value), "tag": tag, "dir": str(ckpt_dir)}
        self.entries.append(meta)
        self._prune_if_needed()
        self._persist()
        return ckpt_dir


# Evaluation logic
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
    """
    A helper function to construct the final `inputs_embeds` tensor for the LLM.

    This is a complex but necessary step because the input to the LLM is a mix of
    text and audio information that is already in the embedding space. It concatenates
    the embeddings for: [Instruction Text] -> [<AUD> Separator] -> [Audio Embeds]
    -> [Target Text (optional)], and handles the required padding and attention masks.
    """
    B = audio_embeds.size(0)
    per_sample_inputs: List[torch.Tensor] = []
    per_sample_labels: List[torch.Tensor] = []
    per_sample_lengths: List[int] = []

    # Process each sample in the batch individually before padding.
    for i in range(B):
        # Embed the instruction prompt.
        prompt_ids = tokenizer(instructions[i], return_tensors="pt").input_ids.to(emb_device)
        prompt_emb = emb_layer(prompt_ids)
        sep_emb = sep_emb_static
        a_emb = audio_embeds[i : i + 1]

        if tgt_ids_batch is not None and tgt_attn_mask is not None:
            # For loss calculation: include target text embeddings (teacher forcing).
            tgt_len = int(tgt_attn_mask[i].sum().item())
            tgt_ids = tgt_ids_batch[i : i + 1, :tgt_len]
            tgt_emb = emb_layer(tgt_ids)
            inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb, tgt_emb], dim=1)
        else:
            # For generation: only include the prefix.
            inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb], dim=1)

        per_sample_inputs.append(inputs_embeds)
        per_sample_lengths.append(int(inputs_embeds.size(1)))

        if tgt_ids_batch is not None and tgt_attn_mask is not None:
            # Create labels for loss calculation, ignoring the prefix part.
            prefix_len = prompt_emb.size(1) + sep_emb.size(1) + a_emb.size(1)
            labels = torch.full(
                (1, inputs_embeds.size(1)), fill_value=-100, dtype=torch.long, device=emb_device
            )
            labels[:, prefix_len:] = tgt_ids
            per_sample_labels.append(labels)

    # Pad all samples in the batch to the same maximum length.
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
    save_predictions_path: Optional[Path] = None,
    num_preview: int = 3,
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Runs the full evaluation loop on a given dataset split (e.g., 'dev').

    It calculates the language modeling loss on the ground-truth summaries and
    also generates new summaries from scratch to see what the model produces.
    Results and previews are logged and can be saved to a file.
    """
    ds = JsonlSummDataset(Path(dataset_jsonl))
    def collate_fn(samples: List[Sample]) -> Tuple[torch.Tensor, Any, List[str], List[str], List[str]]:
        return collate_to_fbank(translator, samples)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    total_loss = 0.0
    total_count = 0
    previews: List[Dict[str, str]] = []

    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        save_predictions_path.write_text("")

    # Set models to evaluation mode.
    llama_was_training = any(p.requires_grad and p.grad is not None for p in llama.parameters()) or llama.training
    llama.eval()

    # It iterates through the evaluation data loader.
    for seqs, padding_mask, targets, insts, audio_paths in tqdm(loader, desc=f"Evaluating {split_name}"):
        seqs = seqs.to(device=device, dtype=dtype)
        if padding_mask is not None and hasattr(padding_mask, "to"):
            padding_mask = padding_mask.to(device)

        # Forward pass through the audio pipeline.
        # For each batch, it performs a forward pass to get the audio embeddings from the Q-Former.
        enc_out, enc_pad = translator.model.encode_speech(seqs, padding_mask)
        enc_out = enc_out.to(dtype=train_dtype)
        audio_tokens = qformer(enc_out, None)
        audio_embeds = projector(audio_tokens).to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

        # Prepare target text tokens.
        tgt_tok = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
        tgt_ids_batch = tgt_tok.input_ids.to(emb_layer.weight.device)
        tgt_attn_mask = tgt_tok.attention_mask.to(emb_layer.weight.device)

        # 1. Calculate validation loss.
        # It calculates the language modeling loss on the ground-truth summaries (to measure how well the model predicts the correct text).
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

        # 2. Generate predictions.
        # It also generates new summaries from scratch using llama.generate to see what the model produces on its own.
        gen_inputs, _, gen_attn, lengths = _build_eval_batch(
            tokenizer,
            emb_layer,
            sep_emb_static,
            audio_embeds,
            insts,
            None, # No target text for generation.
            None,
            emb_layer.weight.device,
        )
        gen_out = llama.generate(
            inputs_embeds=gen_inputs,
            attention_mask=gen_attn,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        # Decode only the newly generated tokens.
        gen_tokens_only = gen_out[:, gen_inputs.size(1):]
        preds = tokenizer.batch_decode(gen_tokens_only, skip_special_tokens=True)

        # Collect previews and save predictions.
        for i in range(min(num_preview - len(previews), bs)):
            previews.append({
                "audio": audio_paths[i],
                "pred": preds[i].strip(),
                "ref": targets[i],
            })
            if len(previews) >= num_preview:
                break

        if save_predictions_path is not None:
            with save_predictions_path.open("a", encoding="utf-8") as f:
                for a, p, r in zip(audio_paths, preds, targets):
                    rec = {"audio": a, "prediction": p.strip(), "reference": r}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_loss = total_loss / max(1, total_count)
    logger.info(f"{split_name} avg_lm_loss={avg_loss:.4f}")

    # Restore LLM to its original training state.
    if llama_was_training:
        llama.train()

    # It returns the average loss and a list of prediction previews.
    return avg_loss, previews


def train(args: argparse.Namespace) -> None:
    """
    The main function that orchestrates the entire training process.
    """
    # --- 1. SETUP ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(name)s: %(message)s")

    # Configure device and data types for training (e.g., CUDA, FP16).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16 = (args.use_fp16 or not args.no_fp16) and torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    amp_enabled = dtype == torch.float16 and device.type == "cuda"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # --- 2. LOAD PRE-TRAINED MODELS ---
    # Load the SeamlessM4T model via its Translator interface. This will be frozen.
    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )
    translator.model.eval()
    set_requires_grad(translator.model, False)

    model_dim = translator.model.model_dim
    # qcfg = AudioQFormerConfig(
    #     model_dim=model_dim,
    #     num_layers=args.qformer_layers,
    #     num_heads=args.qformer_heads,
    #     num_queries=args.qformer_queries,
    #     mlp_ratio=4.0,
    #     dropout=0.1,
    # )

    train_dtype = torch.float32 if amp_enabled else dtype
    qformer = AudioQFormer(qcfg).to(device=device, dtype=train_dtype)
    projector: SimpleProjector

    # Load the LLM (Llama-3-8B-Instruct) and its tokenizer from Hugging Face.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _resolve_llm_repo_id(repo_id: str) -> str:
        # Helper to map short names to full Hugging Face repo IDs.
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

    # Handle Hugging Face authentication for gated models.
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
                if "unexpected keyword argument 'token'" in str(e):
                    return load_fn(model_id, use_auth_token=hf_token, **kwargs)
                raise
        return load_fn(model_id, **kwargs)

    tokenizer = _from_pretrained_with_optional_token(
        AutoTokenizer.from_pretrained, llm_repo_id, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llama = _from_pretrained_with_optional_token(
        AutoModelForCausalLM.from_pretrained,
        llm_repo_id,
        torch_dtype=dtype,
        device_map="auto", # Automatically handle multi-GPU setup.
    )

    # Optionally load LoRA adapters to fine-tune the LLM. Otherwise, freeze it.
    if args.llm_lora is not None and args.llm_lora != "":
        from peft import PeftModel

        llama = PeftModel.from_pretrained(llama, args.llm_lora)
    else:
        set_requires_grad(llama, False)
        llama.eval()

    # Enable gradient checkpointing to save memory during training.
    if args.gradient_checkpointing and hasattr(llama, 'gradient_checkpointing_enable'):
        llama.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")

    # --- 3. INITIALIZE TRAINABLE BRIDGE COMPONENTS ---
    # Creates the instances of AudioQFormer, SimpleProjector, and AlignHeads which will be trained.
    model_dim = translator.model.model_dim # Dimension of SeamlessM4T's encoder output.
    qcfg = AudioQFormerConfig(
        model_dim=model_dim,
        num_layers=args.qformer_layers,
        num_heads=args.qformer_heads,
        num_queries=args.qformer_queries,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    # Use float32 for trainable components during mixed-precision training for stability.
    train_dtype = torch.float32 if amp_enabled else dtype
    qformer = AudioQFormer(qcfg).to(device=device, dtype=train_dtype)

    # Initialize the projector to map Q-Former output to LLM embedding dimension.
    emb_layer = llama.get_input_embeddings()
    llm_emb_dim = emb_layer.embedding_dim
    projector = SimpleProjector(in_dim=model_dim, out_dim=llm_emb_dim).to(device=device, dtype=train_dtype)

    # Initialize the alignment heads and contrastive queues.
    align = AlignHeads(d_q=model_dim, d_t=llm_emb_dim, d_con=getattr(args, "contrastive_dim", 768)).to(device=device, dtype=train_dtype)
    text_queue = ContrastiveQueue(d_con=getattr(args, "contrastive_dim", 768), K=getattr(args, "queue_size", 32768), device=str(device))
    audio_queue = ContrastiveQueue(d_con=getattr(args, "contrastive_dim", 768), K=getattr(args, "queue_size", 32768), device=str(device))

    # Pre-embed the special <AUD> token separator.
    emb_device = emb_layer.weight.device
    llm_dtype = emb_layer.weight.dtype
    sep_ids = tokenizer("<AUD>", add_special_tokens=False, return_tensors="pt").input_ids.to(emb_device)
    with torch.no_grad():
        sep_emb_static = emb_layer(sep_ids)

    # --- 4. SETUP OPTIMIZER, SCHEDULER, AND DATALOADER ---
    # Configures the AdamW optimizer to only update the parameters of the trainable components, sets up a learning rate scheduler, and creates the training DataLoader.

    # Collect all parameters that require gradients for the optimizer.
    params = list(qformer.parameters()) + list(projector.parameters()) + list(align.parameters())
    if any(p.requires_grad for p in llama.parameters()): # Add LoRA params if any.
        params += [p for p in llama.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Setup the data loader for the training set.
    ds = JsonlSummDataset(Path(args.dataset_jsonl))
    def collate_fn(samples: List[Sample]) -> Tuple[torch.Tensor, Any, List[str], List[str], List[str]]:
        return collate_to_fbank(translator, samples)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Setup a cosine learning rate scheduler with a warmup phase.
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

    # Initialize a gradient scaler for stable mixed-precision training.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # --- 5. THE MAIN TRAINING LOOP ---
    # a nested loop over epochs and batches
    step = 0
    # Set trainable modules to training mode.
    qformer.train()
    projector.train()
    align.train()
    llama.train() if any(p.requires_grad for p in llama.parameters()) else llama.eval()

    topk_mgr = TopKCheckpointManager(Path(args.output_dir) / "best", k=args.save_top_k)
    best_dev = float("inf")

    for epoch in range(args.epochs):
        epoch_loader = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for seqs, padding_mask, targets, insts, _audio_paths in epoch_loader:
            step += 1
            # Move data to the appropriate device.
            seqs = seqs.to(device=device, dtype=dtype)
            if padding_mask is not None and hasattr(padding_mask, "to"):
                padding_mask = padding_mask.to(device)

            # --- 5a. Forward pass through the audio encoder and Q-Former ---
            with torch.no_grad():
                # Extract audio features from the frozen SeamlessM4T encoder.
                enc_out, enc_pad = translator.model.encode_speech(seqs, padding_mask)

            # Pass features through the trainable bridge components.
            enc_out = enc_out.to(dtype=train_dtype)
            audio_tokens = qformer(enc_out, None) # Q-Former output.
            audio_embeds = projector(audio_tokens) # Projector output (for LLM).

            # --- 5b. Prepare features for alignment losses ---
            # Pool the Q-Former output to get a single vector representation for the audio.
            q_pool = audio_tokens.mean(dim=1)

            # Get a single vector representation for the corresponding text summary.
            with torch.no_grad():
                tgt_tok = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
                tgt_ids_batch = tgt_tok.input_ids.to(emb_device)
                tgt_attn_mask = tgt_tok.attention_mask.to(emb_device)
                t_input_embeds = emb_layer(tgt_ids_batch)
                mask = tgt_attn_mask.float().unsqueeze(-1)
                t_pool = (t_input_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            # Project audio and text pools into the shared space.
            qz, tz = align.forward_proj(q_pool, t_pool.to(device=device, dtype=train_dtype))

            # --- 5c. Calculate alignment losses ---
            L_con = contrastive_loss_bidir(
                qz,
                tz,
                audio_feats=audio_queue.feats.to(qz.device),
                text_feats=text_queue.feats.to(qz.device),
                logit_scale=align.logit_scale.clamp(0, 5).exp(),
            )
            L_match = matching_loss(align, qz, tz, queue_feats=text_queue.feats.to(qz.device), num_queue_negs=args.num_queue_negs)

            # --- 5d. Construct input and calculate LLM loss ---
            # This is done sample-by-sample and then padded, due to the complexity
            # of concatenating embeddings of different lengths.
            per_sample_inputs: List[torch.Tensor] = []
            per_sample_labels: List[torch.Tensor] = []
            per_sample_lengths: List[int] = []

            for i in range(len(targets)):
                instruction = insts[i]
                prompt_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(emb_device)
                prompt_emb = emb_layer(prompt_ids)

                sep_emb = sep_emb_static

                a_emb = audio_embeds[i : i + 1].to(device=emb_device, dtype=llm_dtype)

                tgt_len = int(tgt_attn_mask[i].sum().item())
                tgt_ids = tgt_ids_batch[i : i + 1, :tgt_len]
                tgt_emb = emb_layer(tgt_ids)

                # Concatenate all parts to form the final input embeddings.
                inputs_embeds = torch.cat([prompt_emb, sep_emb, a_emb, tgt_emb], dim=1)
                per_sample_inputs.append(inputs_embeds)
                per_sample_lengths.append(int(inputs_embeds.size(1)))

                # Create labels for the LLM, ignoring all prefix tokens (instruction, audio).
                prefix_len = prompt_emb.size(1) + sep_emb.size(1) + a_emb.size(1)
                labels = torch.full((1, inputs_embeds.size(1)), fill_value=-100, dtype=torch.long, device=emb_device)
                labels[:, prefix_len:] = tgt_ids
                per_sample_labels.append(labels)
            
            # Pad the batch to the maximum length.
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

            # Get the language modeling loss from the LLM.
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                lm_loss = out.loss

            # --- 5e. Combine losses and backpropagate ---
            # Update contrastive queues with the latest features.
            with torch.no_grad():
                audio_queue.enqueue(qz.detach().to(audio_queue.feats.device))
                text_queue.enqueue(tz.detach().to(text_queue.feats.device))

            # Combine all three loss components with their respective weights.
            total_loss = lm_loss + args.lambda_con * L_con + args.lambda_match * L_match
            # Normalize loss for gradient accumulation.
            loss = total_loss / max(1, args.grad_accum_steps)

            # Perform backward pass using the AMP scaler.
            scaler.scale(loss).backward()

            # --- 5f. Optimizer step (with gradient accumulation) ---
            if step % args.grad_accum_steps == 0:
                # Unscale gradients before clipping.
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                # Optimizer takes a step.
                scaler.step(optimizer)
                # Update the scaler for the next iteration.
                scaler.update()
                # Zero out gradients for the next accumulation cycle.
                optimizer.zero_grad(set_to_none=True)
                # Occasionally clear CUDA cache to free up memory.
                if step % (args.grad_accum_steps * 10) == 0:
                    torch.cuda.empty_cache()
                # Step the learning rate scheduler.
                if scheduler is not None:
                    scheduler.step()

            # --- 5g. Logging and Evaluation ---
            if step % args.log_every == 0:
                # Log the current loss values to the console and progress bar.
                loss_info = (
                    f"epoch={epoch} step={step} loss={(float(loss) * args.grad_accum_steps):.4f} "
                    f"lm={float(lm_loss):.4f} con={float(L_con):.4f} match={float(L_match):.4f}"
                )
                logger.info(loss_info)
                epoch_loader.set_postfix({
                    'loss': f"{(float(loss) * args.grad_accum_steps):.4f}",
                    'lm': f"{float(lm_loss):.4f}",
                    'con': f"{float(L_con):.4f}",
                    'match': f"{float(L_match):.4f}"
                })

            # Periodically run evaluation on the dev set.
            if (step > 0 and step % args.dev_eval_steps == 0 and 
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
                    save_predictions_path=None,
                    num_preview=args.eval_preview_samples,
                )
                best_dev = min(best_dev, dev_loss)
                for j, ex in enumerate(previews):
                    logger.info(f"[dev sample {j+1}] ref={ex['ref'][:200]} | pred={ex['pred'][:200]} | audio={ex['audio']}")
                state = {
                    "qformer": qformer.state_dict(),
                    "projector": projector.state_dict(),
                    "align": align.state_dict(),
                }
                tag = f"epoch{epoch}_step{step}"
                saved_dir = topk_mgr.maybe_save(dev_loss, tag, state)
                if saved_dir is not None:
                    logger.info(f"Saved top-k checkpoint to {saved_dir}")

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
                save_predictions_path=None,
                num_preview=args.eval_preview_samples,
            )
            best_dev = min(best_dev, dev_loss)
            for j, ex in enumerate(previews):
                logger.info(f"[dev sample {j+1}] ref={ex['ref'][:200]} | pred={ex['pred'][:200]} | audio={ex['audio']}")
            state = {
                "qformer": qformer.state_dict(),
                "projector": projector.state_dict(),
                "align": align.state_dict(),
            }
            tag = f"epoch{epoch}_step{step}"
            saved_dir = topk_mgr.maybe_save(dev_loss, tag, state)
            if saved_dir is not None:
                logger.info(f"Saved top-k checkpoint to {saved_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(qformer.state_dict(), out_dir / "qformer.pt")
    torch.save(projector.state_dict(), out_dir / "projector.pt")
    torch.save(align.state_dict(), out_dir / "align_heads.pt")
    logger.info(f"Saved Q-Former, projector, and align heads to {out_dir}")

    if args.test_jsonl is not None and os.path.isfile(args.test_jsonl) and not getattr(args, 'skip_final_test_eval', False):
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
            save_predictions_path=test_pred_path,
            num_preview=args.eval_preview_samples,
        )
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"avg_lm_loss": test_loss}, f, indent=2)
        logger.info(f"Wrote test predictions to {test_pred_path} and metrics to test_metrics.json")
    else:
        logger.info("Skipping final test evaluation. Use --skip_final_test_eval to disable or provide --test_jsonl.")


def build_argparser() -> argparse.ArgumentParser:
    """
    It defines all possible command-line arguments (like learning rate, batch size, model names, number of Q-Former layers, etc.) using argparse, parses them from the user's command
    """
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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps (0=disable)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--use_fp16", action="store_true", help="Use mixed precision training")
    p.add_argument("--no_fp16", action="store_true", help="Disable mixed precision (fp16 enabled by default)")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    p.add_argument("--qformer_layers", type=int, default=2)
    p.add_argument("--qformer_heads", type=int, default=8)
    p.add_argument("--qformer_queries", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./qformer_run1")
    p.add_argument("--dev_jsonl", type=str, default=f"{DATA_DIR}/dev.jsonl")
    p.add_argument("--test_jsonl", type=str, default=f"{DATA_DIR}/test.jsonl")
    p.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size (defaults to train batch size)")
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_preview_samples", type=int, default=3)
    p.add_argument("--save_top_k", type=int, default=10)
    p.add_argument("--lambda_con", type=float, default=0.2, help="Weight for contrastive loss")
    p.add_argument("--lambda_match", type=float, default=0.1, help="Weight for matching loss")
    p.add_argument("--contrastive_dim", type=int, default=768, help="Projection dim for contrastive space")
    p.add_argument("--queue_size", type=int, default=32768, help="Size of contrastive feature queue")
    p.add_argument("--num_queue_negs", type=int, default=64, help="Number of queue negatives for matching loss")
    p.add_argument("--dev_eval_steps", type=int, default=100000, help="Evaluate on dev set every N steps instead of per epoch")
    p.add_argument("--skip_final_test_eval", action="store_true", help="Skip final test evaluation after training (runs by default if --test_jsonl provided)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

