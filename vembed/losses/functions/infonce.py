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
                # Supervised Contrastive Loss: Treat same-label samples as positives
                labels = labels.view(-1, 1)
                mask = torch.eq(labels, labels.T).float()

                # For numerical stability
                logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = logits - logits_max.detach()

                # Denominator: sum of exp(logits) for ALL samples (positives + negatives)
                # Exclude the diagonal (self) from the denominator to follow SupCon Eq 2
                eye = torch.eye(batch_size, device=query_emb.device)
                logits_masked = logits.masked_fill(eye > 0, -1e9)

                exp_logits = torch.exp(logits_masked)
                log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

                # Compute mean of log-likelihood over positive set
                # mask[i, j] = 1 if i and j have same label
                # Exclude the diagonal from the POSITIVE set to avoid optimizing trivial self-similarity
                mask_pos = mask * (1 - eye)

                # Normalize by the number of positives for each anchor (excluding self)
                num_pos = mask_pos.sum(dim=1)
                num_pos = torch.where(num_pos > 0, num_pos, torch.ones_like(num_pos))

                mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / num_pos

                # Only average over anchors that actually had positives
                valid_anchors = mask_pos.sum(dim=1) > 0
                if valid_anchors.sum() > 0:
                    return -mean_log_prob_pos[valid_anchors].mean()
                else:
                    return torch.tensor(0.0, device=logits.device, requires_grad=True)

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
