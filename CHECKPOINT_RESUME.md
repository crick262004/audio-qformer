# Checkpoint and Resume Feature

## Overview
Added full checkpoint saving and resuming capability to `train_qformer_llm.py`. The training script now automatically saves complete training state at the end of each epoch and can resume from any saved checkpoint.

## What Gets Saved

Each checkpoint includes:
- **Model States**: Q-Former, Projector, and Alignment heads
- **Optimizer State**: Adam/AdamW momentum and parameters
- **Scheduler State**: Learning rate schedule progress
- **Training Progress**: Current epoch, global step count, best dev loss
- **Random States**: PyTorch, CUDA, NumPy, and Python RNG states for reproducibility

## Checkpoint Location

Checkpoints are saved to: `<output_dir>/last_checkpoint/`

The checkpoint directory contains:
```
last_checkpoint/
├── qformer.pt          # Q-Former model state
├── projector.pt        # Projector model state  
├── align_heads.pt      # Alignment heads state
├── optimizer.pt        # Optimizer state
├── scheduler.pt        # Learning rate scheduler state
└── metadata.pt         # Training progress and random states
```

## How to Use

### Starting Fresh Training
```bash
python train_qformer_llm.py \
    --output_dir ./my_run \
    --epochs 10 \
    --batch_size 4 \
    ...
```

### Resuming from Checkpoint
```bash
python train_qformer_llm.py \
    --resume_from_checkpoint ./my_run/last_checkpoint \
    --output_dir ./my_run \
    --epochs 10 \
    ...
```

### SLURM Job Resume Example
```bash
#!/bin/bash
#SBATCH --job-name=audio-qformer
#SBATCH --time=24:00:00

OUTPUT_DIR="./qformer_distributed_run"
CHECKPOINT="${OUTPUT_DIR}/last_checkpoint"

# Check if checkpoint exists and add resume flag
RESUME_FLAG=""
if [ -d "$CHECKPOINT" ]; then
    echo "Found checkpoint, resuming training..."
    RESUME_FLAG="--resume_from_checkpoint ${CHECKPOINT}"
fi

srun python train_qformer_llm.py \
    --output_dir ${OUTPUT_DIR} \
    --epochs 50 \
    ${RESUME_FLAG} \
    ...
```

## Behavior

1. **Automatic Saving**: At the end of each epoch, the script saves a checkpoint to `<output_dir>/last_checkpoint/`
2. **Overwrites Previous**: Each epoch overwrites the previous checkpoint, so only one "last_checkpoint" exists
3. **Exact Resumption**: When resuming, training continues from the exact epoch, step, and optimizer state
4. **Distributed Training**: Only rank 0 saves checkpoints, but all ranks can load and synchronize

## Benefits

- **Job Preemption Safety**: If a SLURM job is killed or times out, resume from the last completed epoch
- **Debugging**: Resume training without starting from scratch after fixing bugs
- **Experimentation**: Try different hyperparameters from a specific checkpoint
- **Reproducibility**: Random states are saved, ensuring deterministic resumption

## Notes

- The checkpoint only saves after a **complete epoch**, so work within an interrupted epoch is lost
- The `--save_top_k` best checkpoints (based on dev loss) are still saved separately in `<output_dir>/best/`
- If resuming, ensure other arguments (model architecture, data paths, etc.) match the original run
- Checkpoint size depends on model size (typically a few GB for large models)
