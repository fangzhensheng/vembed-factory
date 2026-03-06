"""Hard Negative Mining loss (in-batch only).

Simple in-batch hard negative mining: selects the top-K hardest negatives
from the current batch based on similarity scores.

This approach is:
- Simple: No additional state to manage
- Efficient: Works well with large batch sizes
- Compatible: Works seamlessly with Gradient Cache and DDP
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("hard_negative")
@LossRegistry.register("in_batch_hard")
class InBatchHardMiningLoss(BaseLoss):
    """In-batch hard negative mining.

    Selects the top-K hardest negatives from the current batch based on
    similarity scores.

    If `use_all_negatives` is True, it behaves like standard InfoNCE.
    If False (default), it uses only the top-K hardest negatives, which
    focuses the gradient on the most difficult samples.

    Args:
        temperature: Temperature for softmax (default: 0.05)
        hard_topk: Number of hard negatives to mine (default: 16)
        use_all_negatives: Whether to use all negatives or just top-k (default: False)
    """

    enable_gather_default: bool = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get("temperature", 0.05)
        # Support both config keys for backward compatibility
        self.hard_topk = config.get("hard_mining_topk", config.get("hard_topk", 16))
        self.use_all_negatives = config.get("use_all_negatives", False)
        self._enable_gather = config.get("enable_gather", self.enable_gather_default)

        self.cross_entropy = nn.CrossEntropyLoss()

    def _mine_hard_negatives(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mine top-K hardest negatives from current batch."""
        batch_size = query_emb.size(0)

        # Compute similarity matrix [B, B]
        sim_matrix = query_emb @ positive_emb.T / self.temperature

        # Create mask to exclude positive pairs (self and same-label)
        neg_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=query_emb.device)
        neg_mask.fill_diagonal_(False)

        if labels is not None:
            labels_view = labels.view(-1, 1)
            same_label = torch.eq(labels_view, labels_view.T)
            neg_mask = neg_mask & ~same_label

        # Mask positives to -inf so they are not selected as hard negatives
        sim_masked = sim_matrix.masked_fill(~neg_mask, -1e9)

        # Find top-K hardest negatives
        # k must be <= actual number of negatives
        num_negatives = neg_mask.sum(dim=1).min().item()
        k = min(self.hard_topk, int(num_negatives))

        if k <= 0:
            # Fallback if no negatives found (e.g. batch size 1)
            return torch.empty(batch_size, 0, device=query_emb.device)

        topk_sim, topk_idx = torch.topk(sim_masked, k=k, dim=1)

        # Gather the negative embeddings corresponding to top-k indices
        flat_indices = topk_idx.view(-1)  # [B*K]
        flat_negatives = positive_emb.index_select(0, flat_indices)  # [B*K, D]
        hard_negatives = flat_negatives.view(batch_size, k, -1)  # [B, K, D]

        return hard_negatives

    def _forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute in-batch hard mining loss."""
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)
        batch_size = query_emb.size(0)

        # Calculate positive similarity: [B, 1]
        pos_sim = (query_emb * positive_emb).sum(dim=1, keepdim=True) / self.temperature

        if self.use_all_negatives:
            # Use all in-batch negatives (Standard InfoNCE logic)
            logits = query_emb @ positive_emb.T / self.temperature
            target = torch.arange(batch_size, device=query_emb.device)
            return self.cross_entropy(logits, target)

        # Hard Mining Path: Only Top-K
        hard_negatives = self._mine_hard_negatives(query_emb, positive_emb, labels)

        if hard_negatives.size(1) == 0:
            # No negatives, return 0 loss
            return torch.tensor(0.0, device=query_emb.device, requires_grad=True)

        # Compute similarity with hard negatives: [B, K]
        neg_sim = (
            torch.bmm(query_emb.unsqueeze(1), hard_negatives.transpose(1, 2)).squeeze(1)
            / self.temperature
        )

        # Concat: [Positive (1) | Hard Negatives (K)]
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # Target is always index 0 (the positive sample)
        target = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)

        return self.cross_entropy(logits, target)
