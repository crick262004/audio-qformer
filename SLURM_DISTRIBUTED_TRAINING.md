# Distributed Training on SLURM Cluster

This guide explains how to run the Audio Q-Former training on a SLURM cluster with multiple nodes and GPUs.

## Overview

The training code has been updated to support **multi-node, multi-GPU distributed training** using PyTorch's DistributedDataParallel (DDP) and NCCL backend.

### What's New

1. **Distributed Training Support**: Added `torch.distributed` support with automatic rank/world_size detection
2. **SLURM Integration**: Detects SLURM environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.)
3. **DistributedDataParallel**: Models are wrapped with DDP for efficient multi-GPU training
4. **DistributedSampler**: Dataset is automatically sharded across all processes
5. **Checkpoint Management**: Only rank 0 saves checkpoints to avoid conflicts
6. **Proper Synchronization**: Barriers ensure all processes wait for checkpoints and evaluations

## Hardware Setup

Your cluster configuration:
- **2 nodes** (machines)
- **2 x A100 GPUs per node**
- **Total: 4 A100 GPUs**

## Quick Start

### 1. Prepare Your Data

Ensure your data files are in JSONL format:
```bash
ls -lh train.jsonl dev.jsonl test.jsonl
```

Each line should contain:
```json
{"audio": "path/to/audio.flac", "summary": "text summary", "tgt_lang": "eng", "instruction": "optional"}
```

### 2. Configure the SLURM Script

Edit `slurm_train_multi_node.sh` to match your cluster:

```bash
# Update these sections:

# 1. Partition name (check with `sinfo`)
#SBATCH --partition=gpu

# 2. Module loads (check with `module avail`)
# module load cuda/12.1
# module load python/3.10

# 3. Environment activation
# source /path/to/your/venv/bin/activate
# OR
# conda activate your_env_name

# 4. Data paths
DATASET_JSONL="./train.jsonl"
DEV_JSONL="./dev.jsonl"
TEST_JSONL="./test.jsonl"

# 5. HuggingFace token (if using gated models like Llama)
# export HF_TOKEN="your_token_here"
```

### 3. Submit the Job

```bash
# Make the script executable
chmod +x slurm_train_multi_node.sh

# Submit to SLURM
sbatch slurm_train_multi_node.sh
```

### 4. Monitor the Job

```bash
# Check job status
squeue -u $USER

# View output logs (replace JOBID with actual job ID)
tail -f logs/train_JOBID.out
tail -f logs/train_JOBID.err

# Cancel job if needed
scancel JOBID
```

## Important Settings

### Batch Size and Gradient Accumulation

With 4 GPUs, the effective batch size is:
```
Effective Batch Size = num_gpus × batch_size × grad_accum_steps
                     = 4 × 1 × 2 = 8
```

Compare this to single-GPU training where you might use:
```
Single GPU: 1 × 1 × 8 = 8  (same effective batch size)
```

**Key point**: Reduce `grad_accum_steps` when using multiple GPUs to maintain the same effective batch size and training dynamics.

### NCCL Settings

The script includes optimized NCCL settings for multi-node training:

```bash
export NCCL_DEBUG=INFO              # Debug output
export NCCL_IB_DISABLE=0            # Enable InfiniBand (if available)
export NCCL_SOCKET_IFNAME=^docker0,lo  # Network interfaces
export NCCL_IB_HCA=mlx5             # InfiniBand adapter
export NCCL_NET_GDR_LEVEL=5         # GPU Direct RDMA
```

**Adjust these based on your cluster's network setup!**

### Memory Optimization

The training uses several memory optimization techniques:
- **Mixed Precision (FP16)**: Enabled by default with `--use_fp16`
- **Gradient Checkpointing**: Enabled by default with `--gradient_checkpointing`
- **Expandable CUDA Segments**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If you encounter OOM errors:
1. Reduce `--batch_size` (currently 1)
2. Reduce `--qformer_queries` (currently 16)
3. Reduce `--qformer_layers` (currently 2)
4. Use a smaller LLM model

## Troubleshooting

### Issue: "Connection timeout" or "NCCL initialization failed"

**Solution**: Check network configuration
```bash
# Test inter-node communication
srun --nodes=2 --ntasks-per-node=1 hostname

# Check InfiniBand (if available)
ibstat

# Adjust NCCL settings in the script
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_SOCKET_IFNAME=eth0  # Use Ethernet interface
```

### Issue: "CUDA out of memory"

**Solution**: Reduce memory usage
```bash
# In slurm_train_multi_node.sh, adjust:
BATCH_SIZE=1              # Already minimal
GRAD_ACCUM_STEPS=1        # Reduce from 2 to 1
# Or modify these in the training arguments:
# --qformer_queries 8     # Reduce from 16
# --qformer_layers 1      # Reduce from 2
```

### Issue: "No such file or directory" for audio files

**Solution**: Ensure data is accessible from all nodes
```bash
# Use shared filesystem (NFS, Lustre, etc.)
# Or copy data to each node's local storage before training

# Check file access from compute nodes:
srun --nodes=2 --ntasks-per-node=1 ls -lh /path/to/data/
```

### Issue: Job runs but training is slow

**Solution**: Check GPU utilization and data loading
```bash
# Monitor GPU usage during training
srun --jobid=JOBID nvidia-smi dmon -s u

# If GPU utilization is low:
# 1. Check if data loading is the bottleneck (increase num_workers)
# 2. Check if network bandwidth is saturated
# 3. Verify NCCL is using InfiniBand (check NCCL_DEBUG output)
```

### Issue: "RuntimeError: DataLoader worker is killed"

**Solution**: Increase memory or reduce workers
```bash
# In the SLURM script:
#SBATCH --mem=256G  # Increase from 128G

# Or in train_qformer_llm.py line 753:
# Change num_workers=0 to num_workers=2 (if enough memory)
```

## Advanced Configuration

### Using Different LLM Models

Edit the SLURM script:
```bash
# Smaller model (less memory)
LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Larger model (more memory, better performance)
LLM_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
```

### Using LoRA for LLM Fine-tuning

If you want to fine-tune the LLM with LoRA:
```bash
# Add this to the python command in slurm_train_multi_node.sh:
--llm_lora /path/to/lora/adapter
```

### Adjusting Evaluation Frequency

```bash
# Step-based evaluation (every N steps)
--dev_eval_steps 100

# Half-epoch evaluation (default)
--dev_eval_steps -1

# Per-epoch evaluation only
--dev_eval_steps 0
```

### Early Stopping

```bash
# Stop if no improvement for 2 epochs (default)
--early_stop_patience 2

# More patient (allow 5 epochs without improvement)
--early_stop_patience 5
```

## Performance Tips

1. **Use InfiniBand if available**: Provides much better inter-node bandwidth than Ethernet
2. **Check data locality**: Store data on fast shared filesystem (Lustre > NFS)
3. **Profile the code**: Use PyTorch profiler to identify bottlenecks
4. **Monitor NCCL**: Set `NCCL_DEBUG=INFO` and check logs for communication issues
5. **Batch size tuning**: Larger batches = better GPU utilization but more memory

## File Structure After Training

```
./qformer_distributed_run/
├── qformer.pt              # Final Q-Former weights
├── projector.pt            # Final projector weights
├── align_heads.pt          # Final alignment heads weights
├── best/                   # Top-K checkpoints by validation loss
│   ├── epoch0_step100_loss2.3456/
│   │   ├── qformer.pt
│   │   ├── projector.pt
│   │   └── align_heads.pt
│   └── ...
├── test_predictions.jsonl  # Generated summaries on test set
└── test_metrics.json       # Test set metrics

./logs/
├── train_JOBID.out         # STDOUT logs
└── train_JOBID.err         # STDERR logs
```

## Resuming Training

The current implementation doesn't support checkpoint resuming out-of-the-box. To add this:

1. Save optimizer and scheduler states
2. Add `--resume_from_checkpoint` argument
3. Load checkpoint and restore training state

Example modification needed in `train_qformer_llm.py` (around line 740):
```python
if args.resume_from_checkpoint:
    checkpoint = torch.load(args.resume_from_checkpoint)
    unwrap_model(qformer).load_state_dict(checkpoint["qformer"])
    unwrap_model(projector).load_state_dict(checkpoint["projector"])
    unwrap_model(align).load_state_dict(checkpoint["align"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_step = checkpoint.get("step", 0)
```

## Differences from Single-GPU Training

| Aspect | Single GPU | Multi-GPU (4 GPUs) |
|--------|-----------|-------------------|
| Training script | Same | Same |
| Device | `cuda:0` | `cuda:{local_rank}` |
| Model | Raw model | DDP-wrapped |
| DataLoader | Shuffle=True | DistributedSampler |
| Effective batch size | 1 × 8 = 8 | 4 × 1 × 2 = 8 |
| Checkpoint saving | Always | Rank 0 only |
| Logging | All output | Rank 0 only |
| Training speed | 1x | ~3.5-3.8x |

Expected speedup: ~3.5-3.8x on 4 GPUs (not perfect 4x due to communication overhead)

## Getting Help

1. **Check SLURM docs**: `man sbatch`, `man srun`
2. **Check cluster wiki**: Your cluster likely has specific documentation
3. **Contact cluster support**: For network, storage, and module issues
4. **PyTorch DDP docs**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
5. **NCCL docs**: https://docs.nvidia.com/deeplearning/nccl/

## Summary

You should now be able to:
- ✅ Run distributed training on 2 nodes with 4 A100 GPUs
- ✅ Submit SLURM jobs and monitor progress
- ✅ Troubleshoot common issues
- ✅ Adjust hyperparameters for your needs
- ✅ Optimize performance for your cluster

Happy training! 🚀
