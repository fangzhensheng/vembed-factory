"""Unit tests for distributed gradient cache functionality.

These tests verify that GradientCache works correctly with DistributedDataParallel (DDP).
The verification script from scripts/verify_grad_cache_distributed.py has been adapted
as a proper pytest test case.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from vembed.core.gradient_cache import GradientCache


class ToyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def setup_ddp(rank, world_size):
    """Setup DDP environment for testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for distributed test")
class TestDistributedGradientCache:
    """Test distributed gradient cache with DDP."""

    def test_grad_cache_matches_ddp_gradients(self):
        """Verify that GradCache produces the same gradients as standard DDP."""

        def run_rank(rank):
            setup_ddp(rank, 2)
            device = torch.device(f"cuda:{rank}")

            torch.manual_seed(42 + rank)
            model = ToyModel().to(device)

            # Save initial state
            init_state = {k: v.clone() for k, v in model.state_dict().items()}

            batch_size = 8
            chunk_size = 2
            x = torch.randn(batch_size, 10, device=device)
            target = torch.randn(batch_size, 5, device=device)

            # --- Standard DDP backward ---
            model_ddp = ToyModel().to(device)
            model_ddp.load_state_dict(init_state)
            model_ddp = DDP(model_ddp, device_ids=[rank])
            model_ddp.zero_grad()

            pred_ddp = model_ddp(x)
            loss_ddp = nn.MSELoss()(pred_ddp, target)
            loss_ddp.backward()

            grads_ddp = {k.replace("module.", ""): v.grad.clone() for k, v in model_ddp.named_parameters() if v.grad is not None}

            # --- GradCache backward (using wrapper API) ---
            model_gc = ToyModel().to(device)
            model_gc.load_state_dict(init_state)
            model_gc = DDP(model_gc, device_ids=[rank])
            model_gc.zero_grad()

            # New wrapper API - loss_fn takes query_emb, positive_emb
            def loss_fn(queries, positives, **kwargs):
                return nn.MSELoss()(queries, kwargs.get("target", positives))

            gc = GradientCache(loss_fn=loss_fn, chunk_size=chunk_size, retrieval_mode="t2t")
            loss_gc = gc.step(
                model_gc,
                {"input_ids": x, "pos_input_ids": target, "target": target},
            )

            grads_gc = {k.replace("module.", ""): v.grad.clone() for k, v in model_gc.named_parameters() if v.grad is not None}

            # --- Verify results ---
            loss_diff = abs(loss_ddp.item() - loss_gc.item())

            max_grad_diff = 0.0
            for k in grads_ddp:
                if k in grads_gc:
                    diff = (grads_ddp[k] - grads_gc[k]).abs().max().item()
                    max_grad_diff = max(max_grad_diff, diff)

            cleanup_ddp()

            assert loss_diff < 1e-4, f"Loss mismatch: {loss_diff}"
            assert max_grad_diff < 1e-4, f"Gradient mismatch: {max_grad_diff}"

        # Run on rank 0
        run_rank(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGradientCacheIntegration:
    """Integration tests for gradient cache with real training scenarios."""

    def test_grad_cache_chunked_forward(self, random_seed):
        """Test that chunked forward produces same results as unchunked."""
        device = torch.device("cuda:0")

        model = ToyModel().to(device)
        loss_fn = nn.MSELoss()

        batch_size = 16
        x = torch.randn(batch_size, 10, device=device)
        target = torch.randn(batch_size, 5, device=device)

        # --- Full batch forward ---
        model.zero_grad()
        pred_full = model(x)
        loss_full = loss_fn(pred_full, target)
        loss_full.backward()
        grads_full = {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}

        # --- Chunked forward with GradCache (new wrapper API) ---
        model.zero_grad()

        def wrapper_loss_fn(queries, positives, **kwargs):
            return loss_fn(queries, positives)

        gc = GradientCache(loss_fn=wrapper_loss_fn, chunk_size=4, retrieval_mode="t2t")
        loss_chunked = gc.step(
            model,
            {"input_ids": x, "pos_input_ids": target},
        )

        grads_chunked = {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}

        # Verify
        assert abs(loss_full - loss_chunked) < 1e-4

        for k in grads_full:
            if k in grads_chunked:
                diff = (grads_full[k] - grads_chunked[k]).abs().max().item()
                assert diff < 1e-4, f"Gradient mismatch for {k}: {diff}"

    def test_grad_cache_with_negatives(self, random_seed):
        """Test GradCache with negative samples."""
        device = torch.device("cuda:0")

        model = ToyModel().to(device)

        def loss_fn_with_neg(queries, positives, negatives=None, **kwargs):
            """InfoNCE-style loss with negatives."""
            q = F.normalize(queries, dim=1)
            p = F.normalize(positives, dim=1)

            pos_sim = (q * p).sum(dim=1, keepdim=True)

            if negatives is not None:
                n = F.normalize(negatives, dim=1)
                if n.dim() == 2:
                    neg_sim = q @ n.T
                else:
                    neg_sim = (q * n).sum(dim=1, keepdim=True)

                logits = torch.cat([pos_sim, neg_sim], dim=1) / 0.07
            else:
                logits = pos_sim / 0.07

            batch_size = q.shape[0]
            labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
            return nn.CrossEntropyLoss()(logits, labels)

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        pos = torch.randn(batch_size, 10, device=device)
        neg = torch.randn(batch_size * 2, 10, device=device)  # 2 negatives per query

        gc = GradientCache(
            loss_fn=loss_fn_with_neg,
            chunk_size=2,
            retrieval_mode="t2t",
        )

        # Should not raise
        loss = gc.step(
            model,
            {"input_ids": x, "pos_input_ids": pos, "neg_input_ids": neg},
        )
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))


@pytest.mark.parametrize("retrieval_mode", ["t2i", "i2i", "i2t", "t2t", "m2i", "m2t"])
def test_grad_cache_retrieval_modes(retrieval_mode, random_seed):
    """Test that GradCache handles different retrieval modes correctly."""
    device = torch.device("cpu")  # Use CPU for faster tests

    model = ToyModel().to(device)

    def simple_loss(queries, positives, **kwargs):
        return ((queries - positives) ** 2).mean()

    gc = GradientCache(loss_fn=simple_loss, chunk_size=2, retrieval_mode=retrieval_mode)

    # Prepare data based on retrieval mode
    batch = {
        "input_ids": torch.randn(4, 10),
        "pos_input_ids": torch.randn(4, 10),
    }

    # Add image inputs for image-based modes
    if "i" in retrieval_mode:  # Has image queries
        batch["query_pixel_values"] = torch.randn(4, 10)

    if retrieval_mode.endswith("i"):  # Has image documents
        batch["pixel_values"] = torch.randn(4, 10)

    # Should not raise
    loss = gc.step(model, batch)
    assert isinstance(loss, float)


def test_grad_cache_empty_batch_handling():
    """Test that GradCache handles edge cases properly."""
    device = torch.device("cpu")

    model = ToyModel().to(device)
    loss_fn = lambda q, p, **kwargs: ((q - p) ** 2).mean()

    gc = GradientCache(loss_fn=loss_fn, chunk_size=2, retrieval_mode="t2t")

    # Very small batch
    batch = {
        "input_ids": torch.randn(2, 10),
        "pos_input_ids": torch.randn(2, 10),
    }

    loss = gc.step(model, batch)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
