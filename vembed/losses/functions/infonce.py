from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("infonce")
class InfoNCELoss(BaseLoss):
    """InfoNCE loss with optional distributed gather support.

    When gather is enabled (in distributed training), embeddings are gathered
    across all processes to compute loss on the global batch, enabling proper
    in-batch negative sampling across multiple GPUs.
    """

    enable_gather_default: bool = True  # InfoNCE defaults to gather enabled

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get("temperature", 0.05)
        self.cross_entropy = nn.CrossEntropyLoss()
        # Enable gather by default for InfoNCE, can be overridden by config
        self._enable_gather = config.get("enable_gather", self.enable_gather_default)

    def _forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        batch_size = query_emb.size(0)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)

        if negative_emb is None:
            # In-batch negatives: every other positive in the batch serves as a negative
            logits = query_emb @ positive_emb.T / self.temperature

            if labels is not None:
                # Mask out same-label samples (Supervised Contrastive Loss style)
                labels = labels.view(-1, 1)
                mask = torch.eq(labels, labels.T).float()

                # Exclude false negatives (same label but different index) from denominator
                eye = torch.eye(batch_size, device=query_emb.device)
                false_negative_mask = (mask - eye) > 0

                logits = logits.masked_fill(false_negative_mask, -1e9)

            target_labels = torch.arange(batch_size, device=query_emb.device)
            return self.cross_entropy(logits, target_labels)

        # Explicit hard negatives
        negative_emb = F.normalize(negative_emb, p=2, dim=1)
        if negative_emb.dim() == 2:
            negative_emb = negative_emb.view(batch_size, -1, negative_emb.size(-1))

        pos_score = (query_emb * positive_emb).sum(dim=1, keepdim=True)
        neg_scores = torch.bmm(query_emb.unsqueeze(1), negative_emb.transpose(1, 2)).squeeze(1)

        logits = torch.cat([pos_score, neg_scores], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
        return self.cross_entropy(logits, labels)
