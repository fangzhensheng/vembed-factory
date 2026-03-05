from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("infonce")
class InfoNCELoss(BaseLoss):
    """InfoNCE loss with optional distributed gather and bidirectional optimization."""

    enable_gather_default: bool = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get("temperature", 0.05)
        self.cross_entropy = nn.CrossEntropyLoss()
        self._enable_gather = config.get("enable_gather", self.enable_gather_default)
        self.loss_bidirectional = config.get("loss_bidirectional", False)

    def _compute_loss_direction(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss for one direction (query → positive)."""
        batch_size = query_emb.size(0)

        # 1. In-batch negatives
        if negative_emb is None:
            logits = query_emb @ positive_emb.T / self.temperature

            if labels is not None:
                return self._compute_supcon_loss(logits, labels, batch_size, query_emb.device)

            target_labels = torch.arange(batch_size, device=query_emb.device)
            return self.cross_entropy(logits, target_labels)

        # 2. Explicit hard negatives
        negative_emb = F.normalize(negative_emb, p=2, dim=-1)

        # Global negatives (shared across batch)
        if negative_emb.dim() == 2:
            pos_sim = (query_emb * positive_emb).sum(dim=1, keepdim=True)
            neg_sim = query_emb @ negative_emb.T
            
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
            target_labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
            return self.cross_entropy(logits, target_labels)

        # Per-query negatives
        if negative_emb.dim() == 3:
            if negative_emb.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch: query {batch_size}, neg {negative_emb.size(0)}")
            
            pos_sim = (query_emb * positive_emb).sum(dim=1, keepdim=True)
            neg_sim = torch.bmm(query_emb.unsqueeze(1), negative_emb.transpose(1, 2)).squeeze(1)
            
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
            target_labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
            return self.cross_entropy(logits, target_labels)
            
        return torch.tensor(0.0, device=query_emb.device, requires_grad=True)

    def _compute_supcon_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        batch_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Compute Supervised Contrastive Loss."""
        labels_view = labels.view(-1, 1)
        mask = torch.eq(labels_view, labels_view.T).float()

        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        eye = torch.eye(batch_size, device=device)
        logits_masked = logits.masked_fill(eye > 0, -1e9)

        exp_logits = torch.exp(logits_masked)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mask_pos = mask * (1 - eye)
        num_pos = mask_pos.sum(dim=1)
        num_pos = torch.where(num_pos > 0, num_pos, torch.ones_like(num_pos))

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / num_pos

        valid_anchors = mask_pos.sum(dim=1) > 0
        if valid_anchors.sum() > 0:
            return -mean_log_prob_pos[valid_anchors].mean()
        
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)

        loss = self._compute_loss_direction(query_emb, positive_emb, negative_emb, labels)

        if self.loss_bidirectional:
            # Reverse direction: Positive -> Query
            # For bidirectional, we use in-batch negatives (symmetric)
            loss_reverse = self._compute_loss_direction(
                positive_emb, query_emb, None, labels
            )
            loss = (loss + loss_reverse) / 2

        return loss
