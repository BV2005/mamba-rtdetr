"""
Minimal helpers for torchrun-launched DistributedDataParallel training.

Usage:
    torchrun --nproc_per_node=<NUM_GPUS> train.py --config ...

torchrun sets RANK / LOCAL_RANK / WORLD_SIZE in the environment for each
spawned process before running the script, which is how setup_distributed()
detects it's running under torchrun vs. a plain `python train.py` call.
"""
import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize the process group if launched via torchrun.

    Returns:
        is_distributed (bool), rank (int), local_rank (int), world_size (int)
    Falls back to (False, 0, 0, 1) for a plain `python train.py` invocation
    (no torchrun), so single-GPU/CPU runs are unaffected.
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

    return True, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank=0):
    return rank == 0


def reduce_mean(value, world_size):
    """Average a Python float across processes (e.g. for logging a loss
    that reflects the whole step, not just this rank's shard of the batch).
    No-op (returns value unchanged) when not running distributed."""
    if world_size < 2:
        return value
    tensor = torch.tensor(value, device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


class NullLogger:
    """No-op stand-in for TrainingLogger on non-main ranks, so trainer code
    doesn't need `if is_main` guards sprinkled around every logger.* call."""
    def log_scalar(self, *a, **k): pass
    def log_scalars(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_lr(self, *a, **k): pass
    def log_model_info(self, *a, **k): pass
    def info(self, *a, **k): pass
    def close(self): pass
