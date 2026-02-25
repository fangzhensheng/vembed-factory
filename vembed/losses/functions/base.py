"""Base class for loss functions with distributed gather support."""

import torch
import torch.nn as nn
from torch import distributed as dist
from typing import Any


class BaseLoss(nn.Module):
    """Base class for loss functions with optional distributed gather support.

    Subclasses should override `_forward` and set `enable_gather_default` to control
    whether gather is enabled by default for that loss type.
    """

    enable_gather_default: bool = False  # Override in subclasses

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self._enable_gather = False

    @property
    def enable_gather(self) -> bool:
        """Check if gather is enabled and possible.

        Gather is only enabled if:
        1. Distributed training is active (world_size > 1)
        2. Gather is enabled for this loss type
        """
        if not self._enable_gather:
            return False

        if not dist.is_available():
            return False

        try:
            return dist.is_initialized() and dist.get_world_size() > 1
        except RuntimeError:
            return False

    def set_gather(self, enabled: bool) -> None:
        """Enable or disable gather for this loss."""
        self._enable_gather = enabled

    def _gather_tensor(self, t: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Gather tensor across all processes via all-gather.

        Args:
            t: Tensor to gather
            axis: Dimension along which to concatenate gathered tensors

        Returns:
            Gathered tensor concatenated along the specified axis.
            If not in distributed mode, returns tensor unchanged.
        """
        if not isinstance(t, torch.Tensor):
            return t

        if not self.enable_gather:
            return t

        t = t.contiguous()

        try:
            # Use differentiable all_gather (allows full-batch gradient)
            from torch.distributed.nn import functional as dist_nn
            gathered_tensors = dist_nn.all_gather(t)
            return torch.cat(gathered_tensors, dim=axis)

        except (ImportError, AttributeError, RuntimeError):
            # Fallback: standard all_gather (no gradient from other ranks)
            try:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            except RuntimeError:
                return t

            gathered = [torch.empty_like(t) for _ in range(world_size)]
            dist.all_gather(gathered, t)
            gathered[rank] = t

            return torch.cat(gathered, dim=axis)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with optional gather support.

        This method handles gather operation before calling the actual loss computation.
        """
        if self.enable_gather:
            # Gather all embeddings to compute loss on global batch
            query_emb = self._gather_tensor(query_emb, axis=0)
            positive_emb = self._gather_tensor(positive_emb, axis=0)

            if negative_emb is not None:
                negative_emb = self._gather_tensor(negative_emb, axis=0)

            if labels is not None:
                labels = self._gather_tensor(labels, axis=0)

        return self._forward(query_emb, positive_emb, negative_emb, labels, **kwargs)

    def _forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Actual loss computation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _forward method")
