import torch
import torch.nn as nn


class MatryoshkaLoss(nn.Module):
    """Wraps any contrastive loss to train at multiple embedding dimensions simultaneously."""

    def __init__(self, loss_fn: nn.Module, dims: list[int], weights: list[float] | None = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.dims = sorted(dims, reverse=True)
        self.weights = weights or [1.0] * len(dims)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        **kwargs,
    ):
        total = 0.0
        for weight, dim in zip(self.weights, self.dims, strict=False):
            q_slice = query_emb[:, :dim]
            p_slice = positive_emb[:, :dim]
            n_slice = None
            if negative_emb is not None:
                n_slice = (
                    negative_emb[:, :dim] if negative_emb.dim() == 2 else negative_emb[:, :, :dim]
                )
            total += weight * self.loss_fn(q_slice, p_slice, n_slice, **kwargs)
        return total
