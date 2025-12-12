#!/bin/bash
#SBATCH --job-name=audio_qformer_eval
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=95G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ============================================
# Audio Q-Former Evaluation Script
# ============================================
# 
# CHECKPOINT: epoch10_step182070_half (best by val loss: 0.9106)
# 
# METRICS COMPUTED:
#   - WER (Word Error Rate)
#   - CER (Character Error Rate)  
#   - ROUGE-1, ROUGE-2, ROUGE-L (summarization quality)
#   - Exact match rate
#   - Length statistics
#
# SAVES:
#   - ALL predictions to predictions.jsonl
#   - Metrics summary to metrics.json
#
# ESTIMATED TIME (1 GPU):
#   - Quick test (500 samples): ~15-20 minutes
#   - Full test (37,299 samples): ~8-10 hours
# ============================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "AUDIO Q-FORMER EVALUATION"
echo "=========================================="
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "=================================="

# Navigate to project directory
cd /nfs_home/users/vintia/rs1/ashin/arnav/audio-qformer-2

# Activate conda environment (same as training)
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate audio-qformer

# ============================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ============================================

# Best checkpoint (by validation LM loss)
CHECKPOINT_DIR="qformer_distributed_run/best/epoch10_step182070_half_loss0.9106"
TEST_JSONL="test_new.jsonl"

# Set to 0 for FULL evaluation, or a number for quick test
MAX_SAMPLES=500  # Quick test first! Change to 0 for full run

# Output directory
if [ "$MAX_SAMPLES" -gt 0 ]; then
    OUTPUT_DIR="eval_results/epoch10_best_quick${MAX_SAMPLES}_fixed"
else
    OUTPUT_DIR="eval_results/epoch10_best_full_fixed"
fi

# Model parameters (MUST match training!)
QFORMER_LAYERS=2
QFORMER_HEADS=8
QFORMER_QUERIES=16
BATCH_SIZE=4
GEN_MAX_TOKENS=128  # References avg ~18 words, no need for 256

# ============================================

TOTAL_SAMPLES=$(wc -l < $TEST_JSONL)
EVAL_SAMPLES=$MAX_SAMPLES
if [ "$MAX_SAMPLES" -eq 0 ]; then
    EVAL_SAMPLES=$TOTAL_SAMPLES
fi

echo ""
echo "=========================================="
echo "EVALUATION CONFIGURATION"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo ""
echo "Test file: $TEST_JSONL"
echo "Total samples in test set: $TOTAL_SAMPLES"
echo "Samples to evaluate: $EVAL_SAMPLES"
echo ""
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Model config:"
echo "  - Q-Former layers: $QFORMER_LAYERS"
echo "  - Q-Former heads: $QFORMER_HEADS" 
echo "  - Q-Former queries: $QFORMER_QUERIES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max gen tokens: $GEN_MAX_TOKENS"
echo ""
echo "Estimated time: ~$((EVAL_SAMPLES / BATCH_SIZE * 3 / 60)) minutes"
echo "=========================================="
echo ""

# Install required packages for metrics
echo "Checking/installing metric packages..."
pip show jiwer > /dev/null 2>&1 || pip install jiwer
pip show rouge-score > /dev/null 2>&1 || pip install rouge-score

echo ""
echo "Starting evaluation at $(date)..."
echo ""

# Run single-GPU evaluation
python eval_checkpoint.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --test_jsonl "$TEST_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --gen_max_new_tokens $GEN_MAX_TOKENS \
    --max_samples $MAX_SAMPLES \
    --qformer_layers $QFORMER_LAYERS \
    --qformer_heads $QFORMER_HEADS \
    --qformer_queries $QFORMER_QUERIES

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo "  - predictions.jsonl (all ${EVAL_SAMPLES} predictions)"
echo "  - metrics.json (WER, CER, ROUGE scores)"
echo ""
echo "Job finished at: $(date)"
echo "=========================================="

