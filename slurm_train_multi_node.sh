#!/bin/bash
#SBATCH --job-name=audio_qformer_train
#SBATCH --nodes=2                    # 2 machines
#SBATCH --ntasks-per-node=2          # 2 GPUs per machine
#SBATCH --gres=gpu:a100:2            # Request 2 A100 GPUs per node
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --mem=128G                   # Memory per node
#SBATCH --time=48:00:00              # Max runtime
#SBATCH --output=logs/train_%j.out   # Output log
#SBATCH --error=logs/train_%j.err    # Error log
#SBATCH --partition=gpu              # GPU partition (adjust for your cluster)

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "=================================="

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load cudnn/8.9
# module load nccl/2.18
# module load python/3.10

# Activate your environment
# source /path/to/your/venv/bin/activate
# OR
# conda activate your_env_name

# Set distributed training environment variables
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# NCCL settings for optimal multi-node performance
export NCCL_DEBUG=INFO

# CRITICAL: Configure network interface for your cluster
# Option 1: If you have InfiniBand (check with `ibstat`)
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5
# export NCCL_NET_GDR_LEVEL=5
# export NCCL_SOCKET_IFNAME=ib0

# Option 2: If you have Ethernet (most common) - CONFIGURED FOR YOUR CLUSTER
export NCCL_IB_DISABLE=1           # Disable InfiniBand, use Ethernet
export NCCL_SOCKET_IFNAME=eno1     # Your cluster network interface (172.24.16.132/24)
                                    # Detected from: ip addr show

# Option 3: Let NCCL auto-detect but exclude virtual/loopback
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0,veth

# Timeout settings (increase if nodes are slow to connect)
export NCCL_SOCKET_TIMEOUT=300000  # 5 minutes timeout

# CUDA settings
# NOTE: Do not set CUDA_VISIBLE_DEVICES - SLURM handles GPU assignment automatically
# SLURM assigns the requested GPUs (2 per node) and sets CUDA_VISIBLE_DEVICES correctly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OMP settings for better CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
LLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="seamlessM4T_v2_large"

# Training hyperparameters
EPOCHS=5
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

# Optional: Set HF token if needed for gated models
# export HF_TOKEN="your_huggingface_token_here"

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
