from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


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


