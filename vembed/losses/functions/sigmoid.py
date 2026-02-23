from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry


@LossRegistry.register("sigmoid")
class SigmoidLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP).

    References:
        https://arxiv.org/abs/2303.15343
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        # Initial values from SigLIP paper/code
        # logit_scale_init_value = 2.6592 (approx ln(14.3))
        # logit_bias_init_value = -10.0
        init_logit_scale = config.get("init_logit_scale", 2.6592)
        init_logit_bias = config.get("init_logit_bias", -10.0)

        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias)))

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Normalize embeddings
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        positive_emb = F.normalize(positive_emb, p=2, dim=-1)

        # Construct candidate documents (positives + optional negatives)
        if negative_emb is not None:
            negative_emb = F.normalize(negative_emb, p=2, dim=-1)
            # Flatten negatives if they are [B, N, D] -> [B*N, D]
            if negative_emb.dim() == 3:
                B_neg, N_neg, D_neg = negative_emb.shape
                negative_emb = negative_emb.view(B_neg * N_neg, D_neg)

            # Concatenate positives (B) and negatives (M)
            all_docs = torch.cat([positive_emb, negative_emb], dim=0)
        else:
            all_docs = positive_emb

        # Compute logits: (B_query, N_docs)
        # logits = (Q @ D^T) * exp(scale) + bias
        logits = torch.matmul(query_emb, all_docs.T) * self.logit_scale.exp() + self.logit_bias

        B = query_emb.size(0)
        N = all_docs.size(0)

        # Create targets: 1 for positives, -1 for negatives
        # Initialize with -1 (negatives)
        targets = torch.full((B, N), -1.0, device=logits.device, dtype=logits.dtype)

        # Set diagonal to 1 (positives) for the first B columns
        # Note: positive_emb corresponds to indices 0..B-1
        targets[:, :B].fill_diagonal_(1.0)

        # Handle label-aware masking / multi-positives
        if labels is not None:
            # labels: [B]
            # Check for same-label pairs within the in-batch positives
            label_col = labels.view(-1, 1)  # [B, 1]
            same_label = label_col.eq(label_col.T)  # [B, B]

            # Where labels match, set target to 1 (false negatives become positives)
            # This handles both the diagonal (already 1) and other same-class samples
            targets[:, :B] = torch.where(
                same_label,
                torch.tensor(1.0, device=targets.device, dtype=targets.dtype),
                targets[:, :B],
            )

        # Compute Sigmoid Loss
        # loss = -mean_over_batch( sum_over_pairs( log_sigmoid(target * logit) ) )
        #      = - (1/B) * sum( log_sigmoid(targets * logits) )
        loss = -F.logsigmoid(targets * logits).sum() / B

        return loss
