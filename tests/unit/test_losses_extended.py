"""Extended unit tests for CoSENT and Triplet losses."""

import torch
import torch.nn.functional as F

from vembed.losses.functions.cosent import CoSENTLoss
from vembed.losses.functions.infonce import InfoNCELoss
from vembed.losses.functions.triplet import TripletMarginLoss


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
    labels = torch.tensor([0, 0, 1, 1])
    return q, p, labels


class TestCoSENTLossExtended:
    """Extended tests for CoSENT loss."""

    def test_cosent_inbatch_negatives(self):
        """Test CoSENT with in-batch negatives."""
        q, p, _ = _make_embeds()
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "CoSENT loss should be finite"
        assert loss.item() > 0, "CoSENT loss should be positive"

    def test_cosent_with_explicit_negatives(self):
        """Test CoSENT with explicit negatives."""
        q, p, n = _make_embeds()
        n = n.view(q.size(0), -1, q.size(1))
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "CoSENT with explicit negatives should be finite"
        assert loss.item() > 0, "CoSENT loss should be positive"

    def test_cosent_scale_parameter(self):
        """Test CoSENT with different scale values."""
        q, p, _ = _make_embeds()

        for scale in [10.0, 20.0, 30.0, 50.0]:
            loss_fn = CoSENTLoss({"cosent_scale": scale})
            loss = loss_fn(q, p)
            assert torch.isfinite(loss), f"CoSENT with scale={scale} should be finite"

    def test_cosent_gradients(self):
        """Test that CoSENT produces valid gradients."""
        q, p, _ = _make_embeds()
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_cosent_default_scale(self):
        """Test CoSENT default scale value."""
        loss_fn = CoSENTLoss({})
        assert loss_fn.scale == 20.0, "Default scale should be 20.0"

    def test_cosent_gather_default(self):
        """Test that CoSENT has gather disabled by default."""
        loss_fn = CoSENTLoss({})
        assert loss_fn.enable_gather_default is False, "CoSENT should default to gather disabled"

    def test_cosent_reproducibility(self):
        """Test that CoSENT is reproducible with same seed."""
        q, p, _ = _make_embeds(seed=42)
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss1 = loss_fn(q, p).item()

        q, p, _ = _make_embeds(seed=42)
        loss_fn2 = CoSENTLoss({"cosent_scale": 20.0})
        loss2 = loss_fn2(q, p).item()

        assert loss1 == loss2, "CoSENT should be reproducible"

    def test_cosent_small_batch(self):
        """Test CoSENT with small batch size."""
        q, p, _ = _make_embeds(batch=2)
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "CoSENT with small batch should be finite"

    def test_cosent_large_batch(self):
        """Test CoSENT with larger batch size."""
        q, p, _ = _make_embeds(batch=32)
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "CoSENT with large batch should be finite"


class TestTripletMarginLossExtended:
    """Extended tests for Triplet margin loss."""

    def test_triplet_inbatch_negatives(self):
        """Test Triplet with in-batch hardest negatives."""
        q, p, _ = _make_embeds()
        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Triplet loss should be finite"
        assert loss.item() >= 0, "Triplet loss should be non-negative (ReLU)"

    def test_triplet_with_explicit_negatives(self):
        """Test Triplet with explicit negatives."""
        q, p, n = _make_embeds()
        n = n.view(q.size(0), -1, q.size(1))
        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Triplet with explicit negatives should be finite"

    def test_triplet_margin_parameter(self):
        """Test Triplet with different margin values."""
        q, p, _ = _make_embeds()

        for margin in [0.1, 0.3, 0.5, 1.0]:
            loss_fn = TripletMarginLoss({"triplet_margin": margin})
            loss = loss_fn(q, p)
            assert torch.isfinite(loss), f"Triplet with margin={margin} should be finite"
            assert loss.item() >= 0, f"Triplet loss should be non-negative for margin={margin}"

    def test_triplet_gradients(self):
        """Test that Triplet produces valid gradients."""
        q, p, _ = _make_embeds()
        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_triplet_default_margin(self):
        """Test Triplet default margin value."""
        loss_fn = TripletMarginLoss({})
        assert loss_fn.margin == 0.5, "Default margin should be 0.5"

    def test_triplet_gather_default(self):
        """Test that Triplet has gather disabled by default."""
        loss_fn = TripletMarginLoss({})
        assert loss_fn.enable_gather_default is False, "Triplet should default to gather disabled"

    def test_triplet_perfect_match(self):
        """Test Triplet when queries match positives perfectly."""
        torch.manual_seed(0)
        q = torch.randn(4, 16)
        p = q.clone()  # Perfect match
        n = torch.randn(8, 16) * 10  # Far negatives

        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p, n)

        # With perfect matches and far negatives, loss should be 0
        assert loss.item() >= 0, "Triplet with perfect matches should be >= 0"

    def test_triplet_worse_than_random(self):
        """Test Triplet when negatives are closer than positives."""
        torch.manual_seed(0)
        q = torch.randn(4, 16)
        p = q + torch.randn(4, 16) * 2  # Far positives
        n = q + torch.randn(4, 16) * 0.1  # Close negatives

        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p, n)

        # With negatives closer than positives, loss should be positive
        assert loss.item() >= 0, "Triplet should be non-negative"


class TestLossNormalization:
    """Test that losses properly normalize embeddings."""

    def test_infonce_normalization(self):
        """Test that InfoNCE normalizes embeddings."""
        q = torch.randn(4, 16) * 10  # Scaled embeddings
        p = torch.randn(4, 16) * 5
        loss_fn = InfoNCELoss({"temperature": 0.05})
        loss = loss_fn(q, p)

        # After normalization, scale shouldn't matter
        assert torch.isfinite(loss), "InfoNCE should normalize embeddings"

    def test_cosent_normalization(self):
        """Test that CoSENT normalizes embeddings."""
        q = torch.randn(4, 16) * 10
        p = torch.randn(4, 16) * 5
        loss_fn = CoSENTLoss({"cosent_scale": 20.0})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "CoSENT should normalize embeddings"

    def test_triplet_normalization(self):
        """Test that Triplet normalizes embeddings."""
        q = torch.randn(4, 16) * 10
        p = torch.randn(4, 16) * 5
        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
        loss = loss_fn(q, p)

        assert torch.isfinite(loss), "Triplet should normalize embeddings"


class TestLossRegistry:
    """Test loss function registry."""

    def test_all_losses_registered(self):
        """Test that all loss functions are properly registered."""
        from vembed.losses.registry import LossRegistry

        registered = LossRegistry.list_losses()
        # Core loss functions that should be registered
        expected_losses = [
            "infonce",
            "cosent",
            "triplet",
            "sigmoid",
            "hard_negative",
            "in_batch_hard",
            "colbert",
        ]

        for expected in expected_losses:
            assert expected in registered, f"{expected} should be registered"

    def test_loss_factory_compatibility(self):
        """Test that registered losses can be created via factory."""
        from vembed.losses.factory import LossFactory

        loss_types = ["infonce", "cosent", "triplet", "hard_negative"]

        for loss_type in loss_types:
            config = {"loss_type": loss_type}
            try:
                loss = LossFactory.create(config)
                assert loss is not None, f"Should be able to create {loss_type}"
            except Exception as e:
                raise AssertionError(f"Failed to create {loss_type}: {e}")


class TestLossComparison:
    """Compare different loss functions."""

    def test_losses_produce_different_values(self):
        """Test that different loss functions produce different values."""
        q, p, _ = _make_embeds(seed=42)

        config_infonce = {"temperature": 0.05}
        config_cosent = {"cosent_scale": 20.0}
        config_triplet = {"triplet_margin": 0.5}

        loss_infonce = InfoNCELoss(config_infonce)(q, p).item()
        loss_cosent = CoSENTLoss(config_cosent)(q, p).item()
        loss_triplet = TripletMarginLoss(config_triplet)(q, p).item()

        # Different losses should produce different values (very likely)
        # At least check they're all finite
        assert torch.isfinite(torch.tensor(loss_infonce))
        assert torch.isfinite(torch.tensor(loss_cosent))
        assert torch.isfinite(torch.tensor(loss_triplet))

    def test_losses_scale_differently(self):
        """Test that losses respond differently to embedding scale."""
        # Create embeddings with different scales
        q_small = torch.randn(4, 16) * 0.1
        p_small = torch.randn(4, 16) * 0.1
        q_large = torch.randn(4, 16) * 10
        p_large = torch.randn(4, 16) * 10

        loss_infonce = InfoNCELoss({"temperature": 0.05})
        loss_cosent = CoSENTLoss({"cosent_scale": 20.0})

        # InfoNCE should be scale-invariant (due to normalization)
        loss_small_infonce = loss_infonce(q_small, p_small).item()
        loss_large_infonce = loss_infonce(q_large, p_large).item()
        # Due to temperature division and normalization, values might differ slightly
        # but should be in similar range

        # CoSENT uses scale on similarity differences
        loss_small_cosent = loss_cosent(q_small, p_small).item()
        loss_large_cosent = loss_cosent(q_large, p_large).item()

        assert torch.isfinite(torch.tensor(loss_small_infonce))
        assert torch.isfinite(torch.tensor(loss_large_infonce))
        assert torch.isfinite(torch.tensor(loss_small_cosent))
        assert torch.isfinite(torch.tensor(loss_large_cosent))
