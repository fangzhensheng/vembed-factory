from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("triplet")
class TripletMarginLoss(BaseLoss):
    """max(0, sim(q, hardest_neg) - sim(q, pos) + margin)

    Gather is disabled by default for Triplet loss.
    """

    enable_gather_default: bool = False

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.margin = config.get("triplet_margin", 0.5)
        self._enable_gather = config.get("enable_gather", self.enable_gather_default)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
    ):
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)

        if negative_emb is not None:
            negative_emb = F.normalize(negative_emb, p=2, dim=1)
            if negative_emb.dim() == 2:
                negative_emb = negative_emb.view(query_emb.size(0), -1, negative_emb.size(-1))

            pos_sim = (query_emb * positive_emb).sum(dim=1)
            neg_sim = torch.bmm(query_emb.unsqueeze(1), negative_emb.transpose(1, 2)).squeeze(1)
            hardest_neg_sim = neg_sim.max(dim=1).values
        else:
            # Mine hardest in-batch negatives from the similarity matrix
            sim_matrix = query_emb @ positive_emb.T
            pos_sim = sim_matrix.diag()

            identity_mask = torch.eye(query_emb.size(0), device=query_emb.device, dtype=torch.bool)
            sim_matrix.masked_fill_(identity_mask, -1e9)
            hardest_neg_sim = sim_matrix.max(dim=1).values

        loss = F.relu(hardest_neg_sim - pos_sim + self.margin)
        return loss.mean()
