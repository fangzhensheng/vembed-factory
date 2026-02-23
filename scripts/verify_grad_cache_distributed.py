import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from vembed.grad_cache.grad_cache import GradCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))

    def forward(self, x):
        return self.net(x)


def run_process(rank, world_size):
    setup(rank, world_size)

    torch.manual_seed(42)
    model = ToyModel().to(rank)

    # Save initial state to ensure both runs start identical
    init_state = model.state_dict()

    # Different seed for data generation per rank
    torch.manual_seed(42 + rank)
    batch_size_per_gpu = 8
    chunk_size = 2  # 4 chunks per GPU
    input_dim = 10

    x = torch.randn(batch_size_per_gpu, input_dim, device=rank, requires_grad=True)
    target = torch.randn(batch_size_per_gpu, 5, device=rank)

    loss_fn = nn.MSELoss()

    if rank == 0:
        logger.info("Running Distributed Baseline (Standard DDP)...")

    model_ddp_base = ToyModel().to(rank)
    model_ddp_base.load_state_dict(init_state)
    model_ddp = DDP(model_ddp_base, device_ids=[rank])
    model_ddp.zero_grad()

    # Forward
    pred_ddp = model_ddp(x)
    loss_ddp = loss_fn(pred_ddp, target)

    # Backward
    loss_ddp.backward()

    # Store gradients
    grads_ddp = {k.replace("module.", ""): v.grad.clone() for k, v in model_ddp.named_parameters()}

    if rank == 0:
        logger.info("Running Distributed GradCache...")

    model_gc_base = ToyModel().to(rank)
    model_gc_base.load_state_dict(init_state)
    model_gc = DDP(model_gc_base, device_ids=[rank])
    model_gc.zero_grad()

    # GradCache expects a loss function that takes *reps and **kwargs
    def gc_loss_fn(*reps, target_val=None):
        return loss_fn(reps[0], target_val)

    gc = GradCache(models=[model_gc], chunk_sizes=chunk_size, loss_fn=gc_loss_fn)

    # Run cache step with no_sync_except_last=True for DDP optimization
    loss_gc = gc.cache_step(x, target_val=target, no_sync_except_last=True)

    # Store gradients
    grads_gc = {k.replace("module.", ""): v.grad.clone() for k, v in model_gc.named_parameters()}

    # Check Loss
    loss_diff = abs(loss_ddp.item() - loss_gc.item())

    # Check Gradients
    max_grad_diff = 0.0
    for k in grads_ddp:
        g_base = grads_ddp[k]
        g_gc = grads_gc[k]
        diff = (g_base - g_gc).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)

    if rank == 0:
        logger.info(f"Rank {rank} | Loss Diff: {loss_diff:.2e}")
        logger.info(f"Rank {rank} | Max Gradient Diff: {max_grad_diff:.2e}")

        if loss_diff < 1e-5 and max_grad_diff < 1e-5:
            logger.info("\nVERIFICATION PASSED: Distributed GradCache matches DDP.")
        else:
            logger.error("\nVERIFICATION FAILED: Mismatches detected.")

    cleanup()


if __name__ == "__main__":
    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs for this test.")
    else:
        mp.spawn(run_process, args=(world_size,), nprocs=world_size, join=True)
