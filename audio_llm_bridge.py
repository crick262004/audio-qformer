"""
This file defines the core architectural components that bridge the gap between a pre-trained
audio encoder (like SeamlessM4T) and a Large Language Model (LLM).

The main components are:
1. AudioQFormer: A transformer-based module that acts as an information bottleneck. It uses a
   small, fixed number of learnable "query tokens" to interact with the rich audio features
   via cross-attention, extracting only the most salient information relevant for summarization.

2. SimpleProjector: A simple linear layer that maps the output of the AudioQFormer into the
   LLM's embedding space, ensuring dimensional compatibility.

Together, these modules create an efficient and lightweight bridge, allowing a powerful LLM to
"understand" and process audio without needing to be fully retrained on audio tasks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AudioQFormerConfig:
    """
    A dataclass to neatly store all hyperparameters for the AudioQFormer model.
    This makes model configuration clean, readable, and easy to manage.
    """
    # The main internal dimension or "width" of the transformer layers.
    model_dim: int = 512
    # The number of stacked CrossAttentionBlock layers in the Q-Former.
    num_layers: int = 2
    # The number of parallel attention heads in each attention mechanism.
    num_heads: int = 8
    # Determines the width of the feed-forward network's hidden layer (model_dim * mlp_ratio).
    mlp_ratio: float = 4.0
    # The fixed number of learnable query tokens, which form the information bottleneck.
    # These tokens are responsible for "querying" the audio features and summarizing them into a fixed-size representation.
    num_queries: int = 32
    # The dropout rate for regularization in attention and MLP layers.
    dropout: float = 0.1


class CrossAttentionBlock(nn.Module):
    """
    A single transformer block implementing cross-attention.

    This block is the fundamental building unit of the AudioQFormer. In each block,
    a set of input `queries` attends to a different set of context inputs, `kv`
    (key-value pairs, which come from the audio encoder). This allows the queries to
    selectively extract and integrate information from the audio features.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        # Layer normalization for the queries.
        self.norm_q = nn.LayerNorm(dim)
        # Separate layer normalization for the key/value pairs (audio features).
        # Normalizing them independently improves training stability.
        self.norm_kv = nn.LayerNorm(dim)
        # The core multi-head attention mechanism, configured for cross-attention.
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # A standard feed-forward network (MLP) that follows the attention step.
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor, kv_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize the queries and the key/value context separately.
        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        # Perform cross-attention: queries attend to the key/value pairs from the audio.
        # In the self.attn call, the queries (q) learn to extract information from the audio features, which serve as both keys (k) and values (k). This is the cross-attention step.
        attn_out, _ = self.attn(q, k, k, key_padding_mask=kv_padding_mask)
        # First residual connection: add the attention output back to the original queries.
        # This helps prevent the vanishing gradient problem and allows the model to learn modifications to the queries rather than transforming them entirely from scratch.
        x = queries + self.dropout(attn_out)
        # Second residual connection: add the MLP output back to the result of the first connection.
        x = x + self.mlp(x)
        return x


class AudioQFormer(nn.Module):
    """
    The Audio Querying Transformer (Q-Former).

    This module distills a long sequence of audio features from an encoder into a short,
    fixed-length sequence of summary embeddings. It does this by passing a set of learnable
    query tokens through several layers of cross-attention(block defined above), where they repeatedly interact
    with the audio features.
    """
    def __init__(self, config: AudioQFormerConfig) -> None:
        super().__init__()
        self.config = config
        # These are the learnable query tokens. They are not input-dependent but are trained
        # to become expert "question askers" that extract salient audio information.
        # It's a learnable nn.Parameter of shape (num_queries, model_dim).
        self.query_tokens = nn.Parameter(torch.randn(config.num_queries, config.model_dim) * 0.02)
        # Create a stack of CrossAttentionBlocks as defined by the config.
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
        # A final layer normalization for the output tokens.
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(self, encoder_output: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get the batch size from the audio encoder's output.
        batch_size = encoder_output.size(0)
        # Prepare the learnable queries for batch processing by expanding them to match the batch size.
        # It expands the self.query_tokens to match the batch size of the input, creating a fresh set of queries for each sample in the batch.
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # Pass the queries through each cross-attention block sequentially.
        # In each block, the queries are refined by attending to the *same* encoder_output.
        x = queries
        for blk in self.blocks:
            x = blk(x, encoder_output, encoder_padding_mask)
        # Apply final normalization and return the refined query tokens. These now represent the audio summary.
        return self.final_norm(x)


class SimpleProjector(nn.Module):
    """
    A simple linear projection layer.

    This module's sole purpose is to map the output dimension of the AudioQFormer
    to the input embedding dimension of the Large Language Model, ensuring they are compatible.

    The SimpleProjector layer actually takes the entire set of query vectors from the Q-Former, not just a single one. If the Q-Former has 16 queries, its output is a sequence of 16 vectors. The projector takes all 16 of these vectors and projects each one into the LLM's embedding space. The LLM then sees this sequence of 16 "audio words" as context.

    the projector's output is used exclusively for the LLM Loss.

    The SimpleProjector acts as a learned translator between the audio and text modalities. Its primary role is to map the output vectors from the AudioQFormer into the LLM's semantic embedding space.

    During training, this module is updated via backpropagation from the LLM's language modeling loss. This process teaches the projector to translate abstract audio representations into vectors that are meaningful and intelligible to the LLM. It effectively learns to place audio concepts into the correct locations within the LLM's pre-existing conceptual map. This translation works in tandem with the AudioQFormer, which co-adapts its output to be more easily translatable.

    The Projector is actually learning the embedding space of the LLM.
    The output of the Projector is really close to what the LLM would embed plain english into.
    (The lm_loss forces the projector to generate a vector for an audio clip of a "rainy day" that lands in the same neighborhood as the LLM's own embeddings for the words "rain," "water," "wet," and "gloomy.)

    The LLM's embedding model is a massive lookup table that can convert thousands of specific text tokens (like "the", "cat", "photosynthesis") into vectors.
    The Projector isn't replacing that. It's more of a specialized translator or an "Audio-to-Embedding" converter. It doesn't know how to embed the word "cat," but it learns how to take the abstract audio summary of a cat meowing from the Q-Former and translate it into a vector that means "cat" to the LLM.    
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # A single linear layer to perform the dimension mapping.
        self.proj = nn.Linear(in_dim, out_dim)
        # Use Xavier Uniform initialization for the weights, a standard practice for stable training.
        nn.init.xavier_uniform_(self.proj.weight)
        # Initialize biases to zero.
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear projection to the input tensor.
        return self.proj(x)


