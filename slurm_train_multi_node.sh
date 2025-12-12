#!/bin/bash
#SBATCH --job-name=audio_qformer_train
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=95G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs
echo "=========================================="
echo "FULL-SCALE TRAINING - 1.1M SAMPLES"
echo "Feature extraction on GPU (main process)"
echo "=========================================="
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
export NCCL_SOCKET_IFNAME=eno1

# Timeout settings (increase if nodes are slow to connect)
export NCCL_SOCKET_TIMEOUT=300000  # 5 minutes timeout

# CUDA settings
# NOTE: Do not set CUDA_VISIBLE_DEVICES - SLURM handles GPU assignment automatically
# SLURM assigns the requested GPUs (2 per node) and sets CUDA_VISIBLE_DEVICES correctly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU settings for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Start GPU monitoring in background (logs every 10 seconds)
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 10 > logs/gpu_util_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
echo "GPU monitoring started (PID: $GPU_MONITOR_PID)"

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
# Optimized for A100 GPUs - increased batch_size for better throughput
# Memory usage is 32-46%, so batch_size=16 should fit comfortably
BATCH_SIZE=16             # Per-GPU batch size (effective batch = 4 * 16 * 1 = 64)
GRAD_ACCUM_STEPS=1        # No accumulation needed
LR=2e-4                   # Learning rate for batch_size=16
WEIGHT_DECAY=0.01

# Q-Former config
QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16

# Evaluation settings - optimized for large dev set (37k samples)
EVAL_BATCH_SIZE=16        # Larger batch for faster evaluation
GEN_MAX_NEW_TOKENS=128
GEN_LIMIT_DEV=10          # Generate on 10 dev samples for monitoring
GEN_LIMIT_TEST=20         # Generate on 20 test samples

# Loss weights
LAMBDA_CON=0.2
LAMBDA_MATCH=0.1

# Display dataset information
echo ""
echo "=========================================="
echo "DATASET INFORMATION:"
echo "Train samples: $(wc -l < $DATASET_JSONL)"
echo "Dev samples: $(wc -l < $DEV_JSONL)"
echo "Test samples: $(wc -l < $TEST_JSONL)"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total GPUs: $SLURM_NTASKS"
echo "Effective batch size: $((BATCH_SIZE * SLURM_NTASKS))"
echo "=========================================="
echo ""

# Check for checkpoint to resume from
CHECKPOINT_DIR="${OUTPUT_DIR}/last_checkpoint"
RESUME_FLAG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "=========================================="
    echo "✓ Found checkpoint at: ${CHECKPOINT_DIR}"
    
    # Display checkpoint info if possible
    python -c "import torch; m = torch.load('${CHECKPOINT_DIR}/metadata.pt', map_location='cpu'); print(f'  Current epoch: {m[\"epoch\"]}'); print(f'  Global step: {m[\"global_step\"]}'); print(f'  Best dev loss: {m[\"best_dev\"]:.4f}')" 2>/dev/null || echo "  Resuming training..."
    
    echo "=========================================="
    echo ""
    RESUME_FLAG="--resume_from_checkpoint ${CHECKPOINT_DIR}"
else
    echo "No checkpoint found, starting training from scratch..."
    echo ""
fi

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
    --early_stop_patience 10 \
    $RESUME_FLAG

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null
echo "GPU monitoring stopped"

echo "Training completed!"
echo "GPU utilization log saved to: logs/gpu_util_${SLURM_JOB_ID}.csv"
