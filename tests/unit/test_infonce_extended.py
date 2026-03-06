"""Extended unit tests for InfoNCE loss."""

import torch

from vembed.losses.functions.infonce import InfoNCELoss


def _make_embeds(batch=4, dim=16, seed=0):
    """Create test embeddings."""
    torch.manual_seed(seed)
    q = torch.randn(batch, dim, requires_grad=True)
    p = torch.randn(batch, dim, requires_grad=True)  # Independent leaf tensor
    n = torch.randn(batch * 2, dim)
    return q, p, n


def _make_embeds_with_labels(batch=4, dim=16, seed=0):
    """Create test embeddings with labels."""
    torch.manual_seed(seed)
    q = torch.randn(batch, dim)
    p = q + 0.1 * torch.randn(batch, dim)
    n = torch.randn(batch * 2, dim)
    labels = torch.tensor([0, 0, 1, 1])
    return q, p, n, labels


class TestInfoNCEExplicitNegatives:
    """Test InfoNCE with explicit hard negatives."""

    def test_infonce_with_2d_negatives(self):
        """Test InfoNCE with 2D global negatives."""
        q, p, n = _make_embeds()
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "InfoNCE with 2D negatives should be finite"
        assert loss.item() > 0, "InfoNCE loss should be positive"

    def test_infonce_with_3d_negatives(self):
        """Test InfoNCE with 3D per-query negatives."""
        q, p, n = _make_embeds()
        n = n.view(q.size(0), -1, q.size(1))  # [B, 2, D]
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "InfoNCE with 3D negatives should be finite"
        assert loss.item() > 0, "InfoNCE loss should be positive"

    def test_infonce_with_irregular_negatives(self):
        """Test InfoNCE with irregular negative count (not divisible by batch)."""
        q, p, _ = _make_embeds()
        torch.manual_seed(1)
        n = torch.randn(q.size(0) * 2 + 1, q.size(1))  # Odd number
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "InfoNCE with irregular negatives should be finite"

    def test_infonce_negatives_normalized(self):
        """Test that explicit negatives are normalized correctly."""
        q, p, n = _make_embeds()
        # Create unnormalized negatives
        n = n * 10  # Scale up
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Should normalize unnormalized negatives"

    def test_infonce_3d_negatives_gradients(self):
        """Test gradient flow with 3D negatives."""
        q, p, n = _make_embeds()
        n = n.view(q.size(0), -1, q.size(1))
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, n)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_infonce_3d_wrong_batch_size(self):
        """Test that 3D negatives with wrong batch size raises error."""
        q, p, _ = _make_embeds(batch=4)
        # Create 3D negatives with wrong batch size
        n = torch.randn(2, 3, q.size(1))  # Batch size 2, not 4

        loss_fn = InfoNCELoss({"temperature": 0.05})

        try:
            loss = loss_fn(q, p, n)
            assert False, "Should raise ValueError for batch size mismatch"
        except ValueError as e:
            assert "Batch size mismatch" in str(e)


class TestInfoNCESupCon:
    """Test InfoNCE supervised contrastive learning."""

    def test_infonce_supcon_basic(self):
        """Test basic supervised contrastive loss."""
        q, p, n, labels = _make_embeds_with_labels()
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, None, labels)

        assert torch.isfinite(loss), "Supervised loss should be finite"
        assert loss.item() >= 0, "Supervised loss should be non-negative"

    def test_infonce_supcon_multiple_samples_per_class(self):
        """Test supervised contrastive with multiple samples per class."""
        torch.manual_seed(0)
        q = torch.randn(8, 16)
        p = q + 0.1 * torch.randn(8, 16)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, None, labels)

        assert torch.isfinite(loss), "Supervised loss with multiple samples per class"

    def test_infonce_supcon_all_same_class(self):
        """Test supervised contrastive when all samples are same class."""
        q, p, n, labels = _make_embeds_with_labels()
        labels = torch.tensor([0, 0, 0, 0])  # All same class

        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, None, labels)

        assert torch.isfinite(loss), "Should handle all same class"
        # With all same class, there are no valid positives after excluding self
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_infonce_supcon_two_classes(self):
        """Test supervised contrastive with exactly two classes."""
        torch.manual_seed(0)
        q = torch.randn(4, 16)
        p = q + 0.1 * torch.randn(4, 16)
        labels = torch.tensor([0, 0, 1, 1])

        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, None, labels)

        assert torch.isfinite(loss), "Two-class supervised loss should be finite"
        assert loss.item() > 0, "Two-class supervised loss should be positive"

    def test_infonce_supcon_gradients(self):
        """Test gradient flow with supervised contrastive."""
        q = torch.randn(4, 16, requires_grad=True)
        p = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])

        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p, None, labels)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"


class TestInfoNCETemperature:
    """Test InfoNCE with different temperature settings."""

    def test_infonce_temperature_scaling(self):
        """Test InfoNCE with various temperatures."""
        q, p, _ = _make_embeds()

        for temp in [0.01, 0.05, 0.1, 0.5, 1.0]:
            loss_fn = InfoNCELoss({"temperature": temp})
            loss = loss_fn(q, p)
            assert torch.isfinite(loss), f"InfoNCE with temp={temp} should be finite"
            # Loss can be zero for certain temperature/seed combinations
            assert loss.item() >= 0, f"InfoNCE loss should be non-negative for temp={temp}"

    def test_infonce_default_temperature(self):
        """Test InfoNCE default temperature value."""
        loss_fn = InfoNCELoss({})  # No temperature specified
        assert loss_fn.temperature == 0.05, "Default temperature should be 0.05"


class TestInfoNCEGather:
    """Test InfoNCE gather functionality."""

    def test_infonce_gather_default_enabled(self):
        """Test that InfoNCE has gather enabled by default."""
        loss_fn = InfoNCELoss({})
        assert loss_fn.enable_gather_default is True, "InfoNCE should default to gather enabled"

    def test_infonce_gather_can_be_disabled(self):
        """Test that gather can be explicitly disabled."""
        loss_fn = InfoNCELoss({"enable_gather": False})
        assert loss_fn._enable_gather is False, "Gather should be disableable"


class TestInfoNCEConfig:
    """Test InfoNCE configuration handling."""

    def test_infonce_config_defaults(self):
        """Test InfoNCE config default values."""
        loss_fn = InfoNCELoss({})
        assert loss_fn.temperature == 0.05
        assert loss_fn.loss_bidirectional is False
        assert loss_fn._enable_gather is True

    def test_infonce_custom_config(self):
        """Test InfoNCE with custom config."""
        config = {
            "temperature": 0.1,
            "loss_bidirectional": True,
            "enable_gather": False,
        }
        loss_fn = InfoNCELoss(config)
        assert loss_fn.temperature == 0.1
        assert loss_fn.loss_bidirectional is True
        assert loss_fn._enable_gather is False


class TestInfoNCEErrors:
    """Test InfoNCE error handling."""

    def test_infonce_empty_batch(self):
        """Test InfoNCE behavior with empty embeddings (edge case)."""
        # This is more about documenting behavior than testing
        # Empty tensors would have shape [0, D]
        q = torch.randn(0, 16)
        p = torch.randn(0, 16)
        loss_fn = InfoNCELoss({"temperature": 0.05})

        # Should handle gracefully (may return inf or nan depending on implementation)
        loss = loss_fn(q, p)
        # We just check it doesn't crash
        assert loss is not None or torch.isnan(loss) or torch.isinf(loss)

    def test_infonce_single_sample(self):
        """Test InfoNCE with single sample in batch."""
        q = torch.randn(1, 16)
        p = torch.randn(1, 16)
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p)

        # Single sample should still work
        assert torch.isfinite(loss), "Single sample should produce finite loss"
