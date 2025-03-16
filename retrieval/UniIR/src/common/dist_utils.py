"""
Distributed Data Parallel utilities
Used for generating dense vector representations of the MBEIR dataset (mbeir_embedder.py)
"""

# Standard Library imports
import os
from datetime import timedelta

# Third-party imports
import torch
import torch.distributed as dist
import math
from torch.utils.data import Sampler


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    # return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, word {}): {}".format(args.rank, args.world_size, args.dist_url),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=60),
    )
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)  # We want to print on all processes


class ContiguousDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples_per_replica = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        # Determine the range of indices for this rank
        start_idx = self.rank * self.num_samples_per_replica
        end_idx = min(start_idx + self.num_samples_per_replica, len(self.dataset))

        # Return the indices for this rank
        return iter(range(start_idx, end_idx))

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch
