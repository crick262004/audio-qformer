#!/bin/bash
# Submit multiple eval jobs for qformer_quick_test_run/best-old checkpoints

cd /nfs_home/users/vintia/rs1/ashin/arnav/audio-qformer-2

CHECKPOINTS=(
  "qformer_quick_test_run/best-old/epoch0_step1109754_loss0.1836:qold_e0_full"
  "qformer_quick_test_run/best-old/epoch1_step2219508_loss0.1613:qold_e1_full"
  "qformer_quick_test_run/best-old/epoch2_step3329262_loss0.1494:qold_e2_full"
  "qformer_quick_test_run/best-old/epoch4_step4993893_half_loss0.1429:qold_e4_half"
)

echo "Submitting evaluation jobs for ${#CHECKPOINTS[@]} quick-old checkpoints..."
echo ""

for entry in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_DIR="${entry%%:*}"
  TAG="${entry##*:}"
  OUTPUT_DIR="eval_results/${TAG}_quick500"

  echo "----------------------------------------"
  echo "Checkpoint: $CHECKPOINT_DIR"
  echo "Output: $OUTPUT_DIR"

  cat > /tmp/eval_${TAG}.sh << EOF2
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
echo "Evaluating quick-old: ${TAG}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

python eval_checkpoint.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --test_jsonl "test_new.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --max_samples 500 \
  --batch_size 4 \
  --gen_max_new_tokens 128 \
  --qformer_layers 2 \
  --qformer_heads 8 \
  --qformer_queries 16

echo "Done quick-old: ${TAG}"
EOF2

  JOB_ID=$(sbatch /tmp/eval_${TAG}.sh | awk '{print $4}')
  echo "Submitted job: $JOB_ID"
  echo ""

done

echo "========================================"
echo "All quick-old jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "========================================"
