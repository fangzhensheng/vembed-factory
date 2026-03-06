"""Unit tests for bidirectional loss functions."""

import torch

from vembed.losses.functions.infonce import InfoNCELoss
from vembed.losses.functions.sigmoid import SigmoidLoss


def _make_embeds(batch=4, dim=16):
    torch.manual_seed(0)
    q = torch.randn(batch, dim)
    p = q + 0.1 * torch.randn(batch, dim)
    n = torch.randn(batch * 2, dim)
    return q, p, n


def _make_embeds_with_labels(batch=4, dim=16):
    torch.manual_seed(0)
    q = torch.randn(batch, dim)
    p = q + 0.1 * torch.randn(batch, dim)
    n = torch.randn(batch * 2, dim)
    labels = torch.tensor([0, 0, 1, 1])
    return q, p, n, labels


class TestInfoNCEBidirectional:
    """Test InfoNCE loss with bidirectional optimization."""

    def test_infonce_unidirectional(self):
        """Test InfoNCE loss in unidirectional mode (default)."""
        q, p, n = _make_embeds()
        config = {"temperature": 0.1, "loss_bidirectional": False}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "InfoNCE unidirectional loss should be finite"
        assert loss.item() > 0, "InfoNCE loss should be positive"

    def test_infonce_bidirectional(self):
        """Test InfoNCE loss in bidirectional mode."""
        q, p, n = _make_embeds()
        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "InfoNCE bidirectional loss should be finite"
        assert loss.item() > 0, "InfoNCE loss should be positive"

    def test_infonce_bidirectional_higher_loss(self):
        """Test that bidirectional loss is different from unidirectional."""
        q, p, n = _make_embeds()
        config_uni = {"temperature": 0.1, "loss_bidirectional": False}
        config_bi = {"temperature": 0.1, "loss_bidirectional": True}

        loss_fn_uni = InfoNCELoss(config_uni)
        loss_fn_bi = InfoNCELoss(config_bi)

        loss_uni = loss_fn_uni(q, p, n).item()
        loss_bi = loss_fn_bi(q, p, n).item()

        assert loss_uni != loss_bi, "Bidirectional and unidirectional losses should differ"

    def test_infonce_bidirectional_with_3d_negatives(self):
        """Test InfoNCE bidirectional with 3D negative embeddings."""
        q, p, n = _make_embeds()
        n = n.view(q.size(0), -1, q.size(1))

        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Bidirectional loss with 3D negatives should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_infonce_bidirectional_with_labels(self):
        """Test InfoNCE bidirectional with supervised contrastive loss."""
        q, p, n, labels = _make_embeds_with_labels()
        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, None, labels)

        assert torch.isfinite(loss), "Bidirectional supervised loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_infonce_bidirectional_config_default(self):
        """Test that bidirectional defaults to False."""
        q, p, n = _make_embeds()
        config = {"temperature": 0.1}  # No loss_bidirectional specified
        loss_fn = InfoNCELoss(config)

        assert loss_fn.loss_bidirectional is False, "Bidirectional should default to False"

    def test_infonce_bidirectional_inbatch_negatives(self):
        """Test InfoNCE bidirectional with in-batch negatives."""
        q, p, _ = _make_embeds()
        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, None)

        assert torch.isfinite(loss), "Bidirectional in-batch negative loss should be finite"
        assert loss.item() > 0, "Loss should be positive"


class TestSigmoidBidirectional:
    """Test Sigmoid loss with bidirectional optimization."""

    def test_sigmoid_unidirectional(self):
        """Test Sigmoid loss in unidirectional mode (default)."""
        q, p, n = _make_embeds()
        config = {"loss_bidirectional": False}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Sigmoid unidirectional loss should be finite"

    def test_sigmoid_bidirectional(self):
        """Test Sigmoid loss in bidirectional mode."""
        q, p, n = _make_embeds()
        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Sigmoid bidirectional loss should be finite"

    def test_sigmoid_bidirectional_different(self):
        """Test that bidirectional loss differs from unidirectional."""
        q, p, n = _make_embeds()
        config_uni = {"loss_bidirectional": False}
        config_bi = {"loss_bidirectional": True}

        loss_fn_uni = SigmoidLoss(config_uni)
        loss_fn_bi = SigmoidLoss(config_bi)

        loss_uni = loss_fn_uni(q, p, n).item()
        loss_bi = loss_fn_bi(q, p, n).item()

        assert loss_uni != loss_bi, "Bidirectional and unidirectional losses should differ"

    def test_sigmoid_bidirectional_with_labels(self):
        """Test Sigmoid bidirectional with labels."""
        q, p, n, labels = _make_embeds_with_labels()
        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n, labels)

        assert torch.isfinite(loss), "Bidirectional loss with labels should be finite"

    def test_sigmoid_bidirectional_inbatch_negatives(self):
        """Test Sigmoid bidirectional with in-batch negatives."""
        q, p, _ = _make_embeds()
        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, None)

        assert torch.isfinite(loss), "Bidirectional in-batch negative loss should be finite"

    def test_sigmoid_bidirectional_config_default(self):
        """Test that bidirectional defaults to False."""
        q, p, n = _make_embeds()
        config = {}
        loss_fn = SigmoidLoss(config)

        assert loss_fn.loss_bidirectional is False, "Bidirectional should default to False"

    def test_sigmoid_bidirectional_with_custom_logit_params(self):
        """Test Sigmoid bidirectional with custom logit parameters."""
        q, p, n = _make_embeds()
        config = {
            "loss_bidirectional": True,
            "init_logit_scale": 3.0,
            "init_logit_bias": -5.0,
        }
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n)

        assert torch.isfinite(loss), "Loss with custom logit params should be finite"


class TestBidirectionalGradients:
    """Test that bidirectional losses produce valid gradients."""

    def test_infonce_bidirectional_gradients(self):
        """Test that bidirectional InfoNCE produces valid gradients."""
        q = torch.randn(4, 16, requires_grad=True)
        p = torch.randn(4, 16, requires_grad=True)
        n = torch.randn(8, 16)

        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)
        loss = loss_fn(q, p, n)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_sigmoid_bidirectional_gradients(self):
        """Test that bidirectional Sigmoid produces valid gradients."""
        q = torch.randn(4, 16, requires_grad=True)
        p = torch.randn(4, 16, requires_grad=True)
        n = torch.randn(8, 16)

        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n)
        loss.backward()

        assert q.grad is not None, "Query should have gradients"
        assert p.grad is not None, "Positive should have gradients"
        assert torch.isfinite(q.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(p.grad).all(), "Positive gradients should be finite"

    def test_sigmoid_logit_params_gradients(self):
        """Test that logit parameters receive gradients."""
        q = torch.randn(4, 16)
        p = torch.randn(4, 16)
        n = torch.randn(8, 16)

        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)
        loss = loss_fn(q, p, n)
        loss.backward()

        assert loss_fn.logit_scale.grad is not None, "logit_scale should have gradients"
        assert loss_fn.logit_bias.grad is not None, "logit_bias should have gradients"


class TestBidirectionalConsistency:
    """Test consistency and correctness of bidirectional losses."""

    def test_infonce_bidirectional_symmetry(self):
        """Test that swapping q and p in bidirectional mode."""
        q, p, n = _make_embeds()
        config = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn = InfoNCELoss(config)

        loss_1 = loss_fn(q, p, n).item()

        loss_fn_2 = InfoNCELoss(config)
        loss_2 = loss_fn_2(p, q, n).item()

        assert (
            abs(loss_1 - loss_2) < 1e-4
        ), f"Swapping q and p should give similar results, got {loss_1} vs {loss_2}"

    def test_sigmoid_bidirectional_no_negatives(self):
        """Test Sigmoid bidirectional with no explicit negatives."""
        q, p, _ = _make_embeds()
        config = {"loss_bidirectional": True}
        loss_fn = SigmoidLoss(config)

        loss = loss_fn(q, p)
        assert torch.isfinite(loss), "Loss with no negatives should be finite"

    def test_bidirectional_batch_size_1(self):
        """Test bidirectional losses with batch size 1."""
        torch.manual_seed(0)
        q = torch.randn(1, 16)
        p = torch.randn(1, 16)
        n = torch.randn(2, 16)

        config_infonce = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn_infonce = InfoNCELoss(config_infonce)
        loss_infonce = loss_fn_infonce(q, p, n)

        config_sigmoid = {"loss_bidirectional": True}
        loss_fn_sigmoid = SigmoidLoss(config_sigmoid)
        loss_sigmoid = loss_fn_sigmoid(q, p, n)

        assert torch.isfinite(loss_infonce), "InfoNCE with batch_size=1 should be finite"
        assert torch.isfinite(loss_sigmoid), "Sigmoid with batch_size=1 should be finite"

    def test_bidirectional_large_batch(self):
        """Test bidirectional losses with larger batch."""
        torch.manual_seed(0)
        q = torch.randn(32, 16)
        p = torch.randn(32, 16)
        n = torch.randn(64, 16)

        config_infonce = {"temperature": 0.1, "loss_bidirectional": True}
        loss_fn_infonce = InfoNCELoss(config_infonce)
        loss_infonce = loss_fn_infonce(q, p, n)

        config_sigmoid = {"loss_bidirectional": True}
        loss_fn_sigmoid = SigmoidLoss(config_sigmoid)
        loss_sigmoid = loss_fn_sigmoid(q, p, n)

        assert torch.isfinite(loss_infonce), "Large batch InfoNCE should be finite"
        assert torch.isfinite(loss_sigmoid), "Large batch Sigmoid should be finite"
