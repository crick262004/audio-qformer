#!/bin/bash
# Submit multiple checkpoint evaluations in parallel

cd /nfs_home/users/vintia/rs1/ashin/arnav/audio-qformer-2

# Checkpoints to evaluate (early to late epochs)
CHECKPOINTS=(
    "qformer_distributed_run/best/epoch7_step130050_half_loss0.9124:epoch7"
    "qformer_distributed_run/best/epoch8_step156060_loss0.9118:epoch8"
    "qformer_distributed_run/best/epoch9_step173400_loss0.9111:epoch9"
    "qformer_distributed_run/best/epoch11_step208080_loss0.9108:epoch11"
)

echo "Submitting evaluation jobs for ${#CHECKPOINTS[@]} checkpoints..."
echo ""

for entry in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_DIR="${entry%%:*}"
    TAG="${entry##*:}"
    OUTPUT_DIR="eval_results/${TAG}_quick500"
    
    echo "----------------------------------------"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "Output: $OUTPUT_DIR"
    
    # Create a temporary SLURM script for this checkpoint
    cat > /tmp/eval_${TAG}.sh << EOF
#!/bin/bash
#SBATCH --job-name=eval_${TAG}
#SBATCH --partition=test-gpu
#SBATCH --qos=test-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=95G
#SBATCH --output=logs/eval_${TAG}_%j.out
#SBATCH --error=logs/eval_${TAG}_%j.err

set -e
mkdir -p logs

cd /nfs_home/users/vintia/rs1/ashin/arnav/audio-qformer-2
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate audio-qformer

echo "=========================================="
echo "Evaluating: ${TAG}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

python eval_checkpoint.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --test_jsonl "test_new.jsonl" \\
    --output_dir "${OUTPUT_DIR}" \\
    --max_samples 500 \\
    --batch_size 4 \\
    --gen_max_new_tokens 128 \\
    --qformer_layers 2 \\
    --qformer_heads 8 \\
    --qformer_queries 16

echo "Done: ${TAG}"
EOF

    # Submit the job
    JOB_ID=$(sbatch /tmp/eval_${TAG}.sh | awk '{print $4}')
    echo "Submitted job: $JOB_ID"
    echo ""
done

echo "========================================"
echo "All ${#CHECKPOINTS[@]} jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "========================================"




