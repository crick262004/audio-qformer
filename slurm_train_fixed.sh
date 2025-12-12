#!/bin/bash
#SBATCH --job-name=qformer_fixed
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=95G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_fixed_%j.out
#SBATCH --error=logs/train_fixed_%j.err

# ============================================
# FIXED TRAINING SCRIPT
# ============================================
# Changes from original:
# 1. Lower learning rate (1e-5 vs 2e-4)
# 2. Added warmup (2000 steps)
# 3. Reduced lambda_con (0.1 vs 0.2)
# 4. Added diversity loss (lambda_div=0.1)
# 5. Improved Q-Former architecture (in audio_llm_bridge.py)
# 6. Normalized projector output (in audio_llm_bridge.py)
# ============================================

set -e

mkdir -p logs
echo "=========================================="
echo "FIXED TRAINING - Audio Q-Former v2"
echo "=========================================="
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
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

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_SOCKET_TIMEOUT=300000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# ============================================
# FIXED HYPERPARAMETERS
# ============================================

# Data paths (update if needed)
DATASET_JSONL="./train.jsonl"  # Update with correct path
DEV_JSONL="./dev.jsonl"        # Update with correct path  
TEST_JSONL="./test.jsonl"      # Update with correct path

# Use a NEW output dir to start fresh
OUTPUT_DIR="./qformer_fixed_run"
LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="seamlessM4T_v2_large"

# FIXED Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
GRAD_ACCUM_STEPS=1

# KEY FIX: Much lower learning rate
LR=1e-5  # Was 2e-4, reduced 20x
WEIGHT_DECAY=0.01
WARMUP_STEPS=2000  # Add warmup

# Q-Former config
QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16

# Evaluation settings
EVAL_BATCH_SIZE=16
GEN_MAX_NEW_TOKENS=128
GEN_LIMIT_DEV=10
GEN_LIMIT_TEST=20

# KEY FIX: Loss weights
LAMBDA_CON=0.1   # Reduced from 0.2 (contrastive was destabilizing)
LAMBDA_MATCH=0.1
LAMBDA_DIV=0.1   # NEW: diversity loss to prevent Q-Former collapse

# ============================================

echo ""
echo "=========================================="
echo "FIXED TRAINING CONFIGURATION"
echo "=========================================="
echo "Output dir: $OUTPUT_DIR"
echo "Learning rate: $LR (was 2e-4)"
echo "Warmup steps: $WARMUP_STEPS"
echo "Lambda_con: $LAMBDA_CON (was 0.2)"
echo "Lambda_div: $LAMBDA_DIV (NEW)"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total GPUs: $SLURM_NTASKS"
echo "Effective batch size: $((BATCH_SIZE * SLURM_NTASKS))"
echo "=========================================="
echo ""

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu_util_fixed_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Run training from SCRATCH (no checkpoint resume)
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
    --lambda_div "$LAMBDA_DIV" \
    --use_fp16 \
    --gradient_checkpointing \
    --save_top_k 10 \
    --early_stop_patience 10

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="




