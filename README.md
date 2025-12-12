# Audio Q-Former: Audio-to-Text Summarization

A lightweight Q-Former fine-tuning framework for audio-to-text summarization, built on top of Meta's SeamlessM4T. This system bridges audio representations from SeamlessM4T to Large Language Models (LLMs) using a Query Transformer (Q-Former) architecture with contrastive alignment learning.

## Architecture

The Audio Q-Former system consists of several key components that work together to transform audio input into high-quality text summaries:

```
Audio Input → SeamlessM4T Encoder → Q-Former → Projector → LLM → Text Summary
     |                                |              |        |
     └─────────────── Contrastive Learning ─────────────────┘
```

### Core Components

1. **SeamlessM4T Audio Encoder**: Converts raw audio to rich multi-modal representations
2. **Q-Former (Query Transformer)**: Cross-attention mechanism that extracts relevant information using learnable queries
3. **Projector**: Linear layer that maps Q-Former outputs to LLM embedding space
4. **Alignment Heads**: Contrastive learning components for audio-text alignment
5. **Large Language Model**: Pre-trained LLM (e.g., Llama) for text generation

### Q-Former Design

The Q-Former is inspired by BLIP-2's approach and consists of:

- **Learnable Query Tokens**: Fixed number of learned embeddings that serve as queries
- **Cross-Attention Blocks**: Multiple layers of cross-attention that attend to audio features
- **Layer Normalization**: Applied to both queries and key-value pairs for stable training

```python
class AudioQFormer(nn.Module):
    def __init__(self, config: AudioQFormerConfig):
        self.query_tokens = nn.Parameter(torch.randn(num_queries, model_dim))
        self.blocks = nn.ModuleList([CrossAttentionBlock(...) for _ in range(num_layers)])
        
    def forward(self, encoder_output, encoder_padding_mask=None):
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, encoder_output, encoder_padding_mask)
        return self.final_norm(queries)
```

### Contrastive Learning Framework

The system employs a sophisticated contrastive learning approach:

#### InfoNCE Contrastive Loss
- **Bidirectional**: Both audio→text and text→audio directions
- **Negative Mining**: Uses in-batch negatives plus maintained queues
- **Temperature Scaling**: Learnable temperature parameter for logit scaling

#### Binary Matching Loss  
- **Positive Pairs**: Aligned audio-text pairs
- **In-Batch Negatives**: Shuffled text within the batch
- **Queue Negatives**: Additional negatives from maintained feature banks

## Data Flow

### Training Pipeline

1. **Audio Loading**: Load audio files and convert to 16kHz waveforms
2. **Feature Extraction**: SeamlessM4T converter creates filterbank features
3. **Q-Former Processing**: Cross-attention extracts relevant information
4. **Projection**: Map to LLM embedding dimension
5. **LLM Forward**: Generate text using teacher forcing
6. **Loss Computation**: Combine language modeling + contrastive losses

### Loss Function
```
Total Loss = λ_lm * LM_Loss + λ_con * Contrastive_Loss + λ_match * Matching_Loss
```

Where:
- **LM_Loss**: Standard cross-entropy for text generation
- **Contrastive_Loss**: InfoNCE for audio-text alignment  
- **Matching_Loss**: Binary classification for pair matching

## Getting Started

### Prerequisites

*   CUDA-capable GPU (recommended for training)
*   Git and Git LFS (for model downloads)

1.  **Create and Activate Conda Environment**

    First, create a new conda environment with Python 3.10:

    ```bash
    conda create -n audio-qformer python=3.10
    conda activate audio-qformer
    ```
    
    Then install librosa
    
    ```
    conda install -c conda-forge libsndfile==1.0.31
    ```

2.  **Clone the Repository**

    ```bash
    git clone https://github.com/ivanj-0/audio-qformer.git
    cd audio-qformer
    ```

3.  **Install Dependencies**

    Install the required packages from the `requirements.finetune.txt` file and then install the project in editable mode.

    ```bash
    pip install -r requirements.finetune.txt
    pip install -e .
    ```

4.  **Authentication Setup (for Gated Models and Llama 3.1 8B Access)**

    To access the gated model, Llama 3.1 8B, you need to authenticate with Hugging Face:

    ```bash
    huggingface-cli login
    ```

    After logging in, make sure you have requested and been granted access to the Llama 3.1 8B model on the [Meta Llama 3.1 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Once approved, you can use the model in your scripts.

5. **Download the Dataset**

	Download the dataset from Hugging Face:
	```bash
	hf download ivanj-0/audio-qformer --repo-type dataset --local-dir "data"
	```
	
	After downloading, unzip all the files. The expected directory structure is as follows:
	
	```
	data/shards
	├── train/
	│   ├── train-000001.json
	│   ├── train-000001.flac
	│   ├── train-000002.json
	│   ├── train-000002.flac
	│   └── ...
	├── dev/
	│   ├── dev-000001.json
	│   ├── dev-000001.flac
	│   └── ...
	└── test/
	    ├── test-000001.json
	    ├── test-000001.flac
	    └── ...
	```
	
	Finally, run `dataset.py` to create `train.jsonl`, `dev.jsonl`, and `test.jsonl` in the root directory.

### Basic Training

To start a basic training session, run the following command:

```bash
python "train_qformer_llm.py" --max_steps 1 --batch_size 1 --grad_accum_steps 1 --log_every 1
```

### Key Training Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--qformer_layers` | Number of Q-Former layers | 2 |
| `--qformer_heads` | Number of attention heads | 8 |
| `--qformer_queries` | Number of learnable queries | 16 |
| `--lambda_con` | Contrastive loss weight | 0.2 |
| `--lambda_match` | Matching loss weight | 0.1 |
| `--use_fp16` | Enable mixed precision | False |
| `--gradient_checkpointing` | Save memory via checkpointing | True |
| `--epochs` | Number of training epochs | 5 |
| `--dev_eval_steps` | Dev eval frequency | -1 (half-epoch) |
| `--early_stop_patience` | Early stopping patience (epochs) | 2 |

### Eval Sampling
- Randomly limit LLM generation during eval to speed up runs, while computing LM loss on the full split.
- Flags:
  - `--gen_limit_dev` (default 3): number of dev examples to generate
  - `--gen_limit_test` (default 10): number of test examples to generate
  - `--gen_seed` (default 42): seed for reproducible sampling
  
Example: `python train_qformer_llm.py --gen_limit_dev 3 --gen_limit_test 10 --gen_seed 42`

### Dev Eval Frequency
- Default: half-epoch and end-of-epoch (`--dev_eval_steps -1`).
- Per-epoch only: `--dev_eval_steps 0`.
- Step-based: `--dev_eval_steps N` (evaluate every N training steps).

## References

This work builds upon several key papers and frameworks:

- **BLIP-2**: [Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- **SeamlessM4T**: [Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)  
- **Q-Former**: Query Transformer architecture for multi-modal understanding
- **Contrastive Learning**: InfoNCE and related methods for representation learning
