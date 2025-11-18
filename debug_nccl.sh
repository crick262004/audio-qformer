#!/bin/bash
#SBATCH --job-name=nccl_debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nccl_debug_%j.out
#SBATCH --error=logs/nccl_debug_%j.err
#SBATCH --partition=gpu

mkdir -p logs

echo "=================================="
echo "NCCL DEBUG TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=================================="

# Set environment
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# CRITICAL: Maximum debug output
export NCCL_DEBUG=TRACE  # Changed from INFO to TRACE for more details
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_SOCKET_TIMEOUT=300000

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Test script
cat > /tmp/nccl_test_$SLURM_JOB_ID.py << 'EOF'
import os
import torch
import torch.distributed as dist

print(f"Process starting...")
print(f"  SLURM_PROCID={os.getenv('SLURM_PROCID')}")
print(f"  SLURM_LOCALID={os.getenv('SLURM_LOCALID')}")
print(f"  SLURM_NODEID={os.getenv('SLURM_NODEID')}")
print(f"  MASTER_ADDR={os.getenv('MASTER_ADDR')}")
print(f"  MASTER_PORT={os.getenv('MASTER_PORT')}")
print(f"  WORLD_SIZE={os.getenv('WORLD_SIZE')}")

# Setup distributed
if "SLURM_PROCID" in os.environ:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    print(f"  Computed: rank={rank}, local_rank={local_rank}, world_size={world_size}")
else:
    print("ERROR: SLURM variables not found!")
    exit(1)

# Set device
torch.cuda.set_device(local_rank)
print(f"  Set CUDA device to {local_rank}")
print(f"  CUDA device count: {torch.cuda.device_count()}")
print(f"  Current device: {torch.cuda.current_device()}")

try:
    print(f"\n[Rank {rank}] Initializing process group...")
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[Rank {rank}] ✓ Process group initialized!")

    print(f"[Rank {rank}] Testing all-reduce...")
    tensor = torch.ones(1).cuda() * rank
    print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")
    dist.all_reduce(tensor)
    print(f"[Rank {rank}] After all-reduce: {tensor.item()} (expected: {sum(range(world_size))})")

    print(f"[Rank {rank}] ✓ NCCL communication working!")
    dist.destroy_process_group()
    print(f"[Rank {rank}] ✓ Test passed!")

except Exception as e:
    print(f"[Rank {rank}] ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

# Run the test
srun python /tmp/nccl_test_$SLURM_JOB_ID.py

echo "=================================="
echo "NCCL DEBUG TEST COMPLETE"
echo "=================================="
