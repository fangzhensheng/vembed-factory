import torch
from torch import nn

from vembed.training.gradient_cache import GradientCache


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        if input_ids is not None:
            return self.fc(input_ids.float())
        if pixel_values is not None:
            return self.fc(pixel_values.float())
        return None


def dummy_loss(q, p, n=None):
    # Simple loss
    loss = ((q - p) ** 2).mean()
    if n is not None:
        loss += ((q - n) ** 2).mean()
    return loss


def test_gradient_cache_step():
    """Test a single step of gradient cache."""
    model = DummyModel()
    loss_fn = dummy_loss
    grad_cache = GradientCache(loss_fn=loss_fn, chunk_size=2, retrieval_mode="t2t")

    # Batch size 4, chunk size 2 -> 2 chunks
    batch = {
        "input_ids": torch.randn(4, 10),
        "attention_mask": torch.ones(4, 10),
        "pos_input_ids": torch.randn(4, 10),
        "pos_attention_mask": torch.ones(4, 10),
        "neg_pixel_values": torch.randn(4, 10),
    }

    # Run step
    loss = grad_cache.step(model, batch)

    assert isinstance(loss, float)
    assert model.fc.weight.grad is not None
