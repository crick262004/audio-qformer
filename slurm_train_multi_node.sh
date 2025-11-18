#!/bin/bash
#SBATCH --job-name=audio_qformer_train
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=95G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Print job info
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "=================================="

# Initialize conda
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate audio-qformer

# Set distributed training environment variables
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# NCCL settings for optimal multi-node performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0,veth

# Timeout settings (increase if nodes are slow to connect)
export NCCL_SOCKET_TIMEOUT=300000  # 5 minutes timeout

# CUDA settings
# NOTE: Do not set CUDA_VISIBLE_DEVICES - SLURM handles GPU assignment automatically
# SLURM assigns the requested GPUs (2 per node) and sets CUDA_VISIBLE_DEVICES correctly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# ============================================
# Training arguments - MODIFY THESE AS NEEDED
# ============================================
DATASET_JSONL="./train.jsonl"
DEV_JSONL="./dev.jsonl"
TEST_JSONL="./test.jsonl"
OUTPUT_DIR="./qformer_distributed_run"
LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="seamlessM4T_v2_large"

# Training hyperparameters
EPOCHS=100
# Optimized for A100 GPUs (can fit batch_size=8 per GPU)
# Option A: Match single-GPU effective batch (recommended for comparing results)
BATCH_SIZE=2              # Per-GPU batch size
GRAD_ACCUM_STEPS=1        # No accumulation needed (effective batch = 4 * 2 * 1 = 8)
# Option B: Maximize throughput (uncomment to use, adjust LR to 2e-4)
# BATCH_SIZE=8            # Max per A100 (effective batch = 4 * 8 * 1 = 32)
# GRAD_ACCUM_STEPS=1
LR=1e-4                   # Use 2e-4 if using batch_size=8
WEIGHT_DECAY=0.01

# Q-Former config
QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16

# Evaluation settings
EVAL_BATCH_SIZE=2
GEN_MAX_NEW_TOKENS=128
GEN_LIMIT_DEV=3
GEN_LIMIT_TEST=10

# Loss weights
LAMBDA_CON=0.2
LAMBDA_MATCH=0.1

# Run the distributed training
srun python train_qformer_llm.py \
    --dataset_jsonl "$DATASET_JSONL" \
    --dev_jsonl "$DEV_JSONL" \
    --test_jsonl "$TEST_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --llm_model_name "$LLM_MODEL" \
    --model_name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --qformer_layers "$QFORMER_LAYERS" \
    --qformer_heads "$QFORMER_HEADS" \
    --qformer_queries "$QFORMER_QUERIES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gen_max_new_tokens "$GEN_MAX_NEW_TOKENS" \
    --gen_limit_dev "$GEN_LIMIT_DEV" \
    --gen_limit_test "$GEN_LIMIT_TEST" \
    --lambda_con "$LAMBDA_CON" \
    --lambda_match "$LAMBDA_MATCH" \
    --use_fp16 \
    --gradient_checkpointing \
    --save_top_k 10 \
    --early_stop_patience 2

echo "Training completed!"
