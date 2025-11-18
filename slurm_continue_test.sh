#!/bin/bash
#SBATCH --job-name=qformer_continue
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=95G
#SBATCH --output=logs/continue_%j.out
#SBATCH --error=logs/continue_%j.err
#SBATCH --time=00:30:00

# Create logs directory if it doesn't exist
mkdir -p logs
echo "==================================="
echo "CONTINUING FROM CHECKPOINT - EPOCH 2"
echo "==================================="
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
export MASTER_PORT=29501  # Use different port to avoid conflicts
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# NCCL settings for optimal multi-node performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_SOCKET_TIMEOUT=300000  # 5 minutes timeout

# CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU settings for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# ============================================
# CONTINUE TRAINING FROM CHECKPOINT
# ============================================
DATASET_JSONL="./train_quick_test.jsonl"  # 20 samples
DEV_JSONL="./dev_quick_test.jsonl"        # 5 samples
TEST_JSONL="./test_quick_test.jsonl"      # 3 samples
OUTPUT_DIR="./qformer_quick_test_run"
LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="seamlessM4T_v2_large"

# Training hyperparameters - SAME AS BEFORE
EPOCHS=5                  # Continue to epoch 5 (3 more epochs)
BATCH_SIZE=4              
GRAD_ACCUM_STEPS=1        
LR=2e-4
WEIGHT_DECAY=0.01

# Q-Former config
QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16

# Evaluation settings
EVAL_BATCH_SIZE=4         
GEN_MAX_NEW_TOKENS=64     
GEN_LIMIT_DEV=2           
GEN_LIMIT_TEST=2

# Loss weights
LAMBDA_CON=0.2
LAMBDA_MATCH=0.1

# Check for checkpoint
CHECKPOINT_DIR="${OUTPUT_DIR}/last_checkpoint"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo ""
    echo "==================================="
    echo "✓ Found checkpoint at: ${CHECKPOINT_DIR}"
    
    # Check metadata
    python -c "import torch; m = torch.load('${CHECKPOINT_DIR}/metadata.pt', map_location='cpu'); print(f'  Current epoch: {m[\"epoch\"]}'); print(f'  Global step: {m[\"global_step\"]}'); print(f'  Best dev loss: {m[\"best_dev\"]:.4f}')" 2>/dev/null || echo "  (Could not read metadata)"
    
    echo "  Resuming training..."
    echo "==================================="
    echo ""
    RESUME_FLAG="--resume_from_checkpoint ${CHECKPOINT_DIR}"
else
    echo "ERROR: No checkpoint found at ${CHECKPOINT_DIR}"
    echo "Please run the initial training first!"
    exit 1
fi

echo ""
echo "==================================="
echo "TRAINING CONTINUATION INFO:"
echo "Train samples: $(wc -l < $DATASET_JSONL)"
echo "Dev samples: $(wc -l < $DEV_JSONL)"
echo "Test samples: $(wc -l < $TEST_JSONL)"
echo "Total epochs: $EPOCHS (will train 3 more epochs)"
echo "Batch size: $BATCH_SIZE"
echo "==================================="
echo ""

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
    --save_top_k 3 \
    --early_stop_patience 5 \
    $RESUME_FLAG

echo ""
echo "==================================="
echo "Training continuation completed!"
echo "Check results in: $OUTPUT_DIR"
echo "Job finished on: $(date)"
echo "==================================="
