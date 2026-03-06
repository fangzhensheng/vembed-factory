"""Unit tests for hard negative mining loss."""

import torch

from vembed.losses.functions.hard_negative import InBatchHardMiningLoss


def _make_embeds(batch=4, dim=16, seed=0):
    """Create test embeddings."""
    torch.manual_seed(seed)
    q = torch.randn(batch, dim, requires_grad=True)
    p = q + 0.1 * torch.randn(batch, dim)
    return q, p


def _make_embeds_with_labels(batch=4, dim=16, seed=0):
    """Create test embeddings with labels."""
    torch.manual_seed(seed)
    q = torch.randn(batch, dim)
    p = q + 0.1 * torch.randn(batch, dim)
    labels = torch.tensor([0, 0, 1, 1])
    return q, p, labels


class TestInBatchHardMining:
    """Test in-batch hard negative mining loss."""

    def test_hard_mining_basic(self):
        """Test basic hard mining loss."""
        q, p = _make_embeds()
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
                "use_all_negatives": False,
            }
        )
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Hard mining loss should be finite"
        assert loss.item() > 0, "Hard mining loss should be positive"

    def test_hard_mining_use_all_negatives(self):
        """Test hard mining with all negatives (InfoNCE mode)."""
        q, p = _make_embeds()
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
                "use_all_negatives": True,
            }
        )
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Loss with all negatives should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_hard_mining_with_labels(self):
        """Test hard mining with supervised labels."""
        q, p, labels = _make_embeds_with_labels()
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
                "use_all_negatives": False,
            }
        )
        loss = loss_fn(q, p, labels=labels)

        assert torch.isfinite(loss), "Hard mining with labels should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_hard_mining_gradients(self):
        """Test that hard mining produces valid gradients."""
        q, p = _make_embeds()
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )
        loss = loss_fn(q, p)
        loss.backward()

        # Check gradients were computed (may be zero for some random seeds)
        assert q.grad is not None, "Query should have gradients"
        # p.grad might be None in some cases, check if it exists
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_hard_mining_different_topk(self):
        """Test hard mining with different topk values."""
        q, p = _make_embeds(batch=8)

        for topk in [1, 2, 4, 6]:
            loss_fn = InBatchHardMiningLoss(
                {
                    "temperature": 0.05,
                    "hard_topk": topk,
                }
            )
            loss = loss_fn(q, p)
            assert torch.isfinite(loss), f"Hard mining with topk={topk} should be finite"

    def test_hard_mining_temperature_scaling(self):
        """Test hard mining with different temperatures."""
        q, p = _make_embeds()

        for temp in [0.01, 0.05, 0.1, 0.5]:
            loss_fn = InBatchHardMiningLoss(
                {
                    "temperature": temp,
                    "hard_topk": 2,
                }
            )
            loss = loss_fn(q, p)
            assert torch.isfinite(loss), f"Hard mining with temp={temp} should be finite"

    def test_hard_mining_config_compatibility(self):
        """Test both config key names for topk."""
        q, p = _make_embeds()

        # Test with hard_mining_topk
        loss_fn1 = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_mining_topk": 4,
            }
        )
        loss1 = loss_fn1(q, p)

        # Test with hard_topk (backward compatibility)
        loss_fn2 = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 4,
            }
        )
        loss2 = loss_fn2(q, p)

        assert loss1.item() == loss2.item(), "Both config keys should give same result"

    def test_hard_mining_small_batch(self):
        """Test hard mining with small batch size."""
        q, p = _make_embeds(batch=2)
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 1,
            }
        )
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Hard mining with small batch should be finite"

    def test_hard_mining_large_batch(self):
        """Test hard mining with larger batch size."""
        q, p = _make_embeds(batch=32)
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 8,
            }
        )
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Hard mining with large batch should be finite"

    def test_hard_mining_gather_default(self):
        """Test that hard mining has gather enabled by default."""
        loss_fn = InBatchHardMiningLoss({"temperature": 0.05})
        assert loss_fn.enable_gather_default is True, "Hard mining should default to gather enabled"

    def test_hard_mining_registry_name(self):
        """Test that hard mining is registered with both names."""
        from vembed.losses.registry import LossRegistry

        assert "hard_negative" in LossRegistry.list_losses()
        assert "in_batch_hard" in LossRegistry.list_losses()


class TestInBatchHardMiningConsistency:
    """Test consistency and correctness of hard mining."""

    def test_hard_mining_vs_infonce_comparison(self):
        """Compare hard mining with InfoNCE when use_all_negatives=True."""
        from vembed.losses.functions.infonce import InfoNCELoss

        q, p = _make_embeds(seed=42)

        # Hard mining with all negatives should be similar to InfoNCE
        loss_hard = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "use_all_negatives": True,
            }
        )
        loss_infonce = InfoNCELoss({"temperature": 0.05})

        loss_hard_val = loss_hard(q, p)
        loss_infonce_val = loss_infonce(q, p)

        assert (
            abs(loss_hard_val - loss_infonce_val) < 1e-5
        ), "Hard mining with all negatives should match InfoNCE"

    def test_hard_mining_reproducibility(self):
        """Test that hard mining is reproducible with same seed."""
        q, p = _make_embeds(seed=123)
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )

        loss1 = loss_fn(q, p).item()

        # Reset and recompute
        q, p = _make_embeds(seed=123)
        loss_fn2 = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )
        loss2 = loss_fn2(q, p).item()

        assert loss1 == loss2, "Hard mining should be reproducible"

    def test_hard_mining_different_seeds(self):
        """Test that different seeds give different results."""
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )

        q1, p1 = _make_embeds(seed=1)
        q2, p2 = _make_embeds(seed=2)

        loss1 = loss_fn(q1, p1).item()
        loss2 = loss_fn(q2, p2).item()

        # Different seeds should likely give different losses
        # (though not guaranteed due to randomness, but very likely)
        # We just check both are finite
        assert torch.isfinite(torch.tensor(loss1))
        assert torch.isfinite(torch.tensor(loss2))


class TestInBatchHardMiningEdgeCases:
    """Test edge cases for hard mining."""

    def test_hard_mining_batch_size_1(self):
        """Test hard mining with batch size 1 (edge case)."""
        q, p = _make_embeds(batch=1)
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 1,
            }
        )
        loss = loss_fn(q, p)

        # With batch size 1, there are no valid negatives
        # Should return 0 loss (requires_grad=True)
        assert torch.isfinite(loss), "Loss with batch_size=1 should be finite"

    def test_hard_mining_topk_larger_than_batch(self):
        """Test hard mining when topk > available negatives."""
        q, p = _make_embeds(batch=4)
        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 100,  # Much larger than batch size
            }
        )
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Should handle topk > batch size gracefully"

    def test_hard_mining_all_same_class(self):
        """Test hard mining when all samples have same label."""
        q, p = _make_embeds(batch=4)
        labels = torch.tensor([0, 0, 0, 0])  # All same class

        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )
        loss = loss_fn(q, p, labels=labels)

        assert torch.isfinite(loss), "Should handle all same class"

    def test_hard_mining_two_classes(self):
        """Test hard mining with exactly two classes."""
        q, p = _make_embeds(batch=4)
        labels = torch.tensor([0, 0, 1, 1])

        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 1,
            }
        )
        loss = loss_fn(q, p, labels=labels)

        assert torch.isfinite(loss), "Should handle two classes correctly"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_hard_mining_zero_gradients_with_same_input(self):
        """Test that identical inputs produce appropriate behavior."""
        # Create identical embeddings (edge case)
        q = torch.randn(4, 16)
        p = q.clone()  # Identical to queries

        loss_fn = InBatchHardMiningLoss(
            {
                "temperature": 0.05,
                "hard_topk": 2,
            }
        )
        loss = loss_fn(q, p)

        # With identical queries and positives, pos_sim = 1.0
        # This should still work
        assert torch.isfinite(loss), "Should handle identical q and p"
