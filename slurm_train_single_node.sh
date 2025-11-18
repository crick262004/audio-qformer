#!/bin/bash
#SBATCH --job-name=audio_qformer_single
#SBATCH --nodes=1                    # SINGLE NODE for testing
#SBATCH --ntasks-per-node=2          # 2 GPUs
#SBATCH --gres=gpu:a100:2            # Request 2 A100 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

echo "=================================="
echo "SINGLE-NODE TRAINING TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: 2"
echo "=================================="

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.10

# Activate environment
# source /path/to/venv/bin/activate
# conda activate your_env_name

# Set distributed training environment variables
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# NCCL settings (simpler for single node)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # Disable InfiniBand
export NCCL_SOCKET_IFNAME=lo       # Use loopback for single node (faster)

# CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Using loopback interface for single-node communication"

# Training arguments
DATASET_JSONL="./train.jsonl"
DEV_JSONL="./dev.jsonl"
TEST_JSONL="./test.jsonl"
OUTPUT_DIR="./qformer_single_node_run"
LLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="seamlessM4T_v2_large"

# Reduced for testing (2 GPUs instead of 4)
BATCH_SIZE=4              # 2 GPUs × 4 batch = 8 effective
GRAD_ACCUM_STEPS=1
LR=1e-4
WEIGHT_DECAY=0.01

QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16

EVAL_BATCH_SIZE=2
GEN_MAX_NEW_TOKENS=128
GEN_LIMIT_DEV=3
GEN_LIMIT_TEST=10

LAMBDA_CON=0.2
LAMBDA_MATCH=0.1

# Check for checkpoint to resume from
CHECKPOINT_DIR="${OUTPUT_DIR}/last_checkpoint"
RESUME_FLAG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Found checkpoint at ${CHECKPOINT_DIR}, resuming training..."
    RESUME_FLAG="--resume_from_checkpoint ${CHECKPOINT_DIR}"
else
    echo "No checkpoint found, starting training from scratch..."
fi

# Run training
srun python train_qformer_llm.py \
    --dataset_jsonl "$DATASET_JSONL" \
    --dev_jsonl "$DEV_JSONL" \
    --test_jsonl "$TEST_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --llm_model_name "$LLM_MODEL" \
    --model_name "$MODEL_NAME" \
    --epochs 5 \
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
    --early_stop_patience 2 \
    $RESUME_FLAG

echo "Training completed!"
