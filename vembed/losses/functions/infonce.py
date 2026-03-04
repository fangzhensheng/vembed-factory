from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("infonce")
class InfoNCELoss(BaseLoss):
    """InfoNCE loss with optional distributed gather and bidirectional optimization.

    When gather is enabled (in distributed training), embeddings are gathered
    across all processes to compute loss on the global batch, enabling proper
    in-batch negative sampling across multiple GPUs.

    When bidirectional is enabled, optimizes both query→positive and positive→query
    directions equally, leading to better alignment for retrieval tasks.
    """

    enable_gather_default: bool = True  # InfoNCE defaults to gather enabled

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

        if negative_emb is None:
            # In-batch negatives
            logits = query_emb @ positive_emb.T / self.temperature

            if labels is not None:
                # Supervised Contrastive Loss
                labels_view = labels.view(-1, 1)
                mask = torch.eq(labels_view, labels_view.T).float()

                # Numerical stability
                logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = logits - logits_max.detach()

                eye = torch.eye(batch_size, device=query_emb.device)
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
                else:
                    return torch.tensor(0.0, device=logits.device, requires_grad=True)

            target_labels = torch.arange(batch_size, device=query_emb.device)
            return self.cross_entropy(logits, target_labels)

        # Explicit hard negatives
        negative_emb_norm = F.normalize(negative_emb, p=2, dim=-1)

        if negative_emb_norm.dim() == 2:
            neg_rows = negative_emb_norm.size(0)

            # If negatives are provided as a flat list (e.g., concatenated across the batch),
            # treat them as global/shared negatives.
            if neg_rows % batch_size != 0:
                all_docs = torch.cat([positive_emb, negative_emb_norm], dim=0)
                logits = query_emb @ all_docs.T / self.temperature
                target_labels = torch.arange(batch_size, device=query_emb.device)
                return self.cross_entropy(logits, target_labels)

            negative_emb_norm = negative_emb_norm.view(batch_size, -1, negative_emb_norm.size(-1))

        if negative_emb_norm.size(0) != batch_size:
            raise ValueError(
                f"negative_emb has incompatible batch dimension: "
                f"query batch={batch_size}, negative batch={negative_emb_norm.size(0)}"
            )

        pos_score = (query_emb * positive_emb).sum(dim=1, keepdim=True)
        neg_scores = torch.bmm(query_emb.unsqueeze(1), negative_emb_norm.transpose(1, 2)).squeeze(1)

        logits = torch.cat([pos_score, neg_scores], dim=1) / self.temperature
        target_labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
        return self.cross_entropy(logits, target_labels)

    def _forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)

        loss_forward = self._compute_loss_direction(query_emb, positive_emb, negative_emb, labels)

        if self.loss_bidirectional:
            loss_reverse = self._compute_loss_direction(
                positive_emb, query_emb, negative_emb, labels
            )
            return (loss_forward + loss_reverse) / 2

        return loss_forward
