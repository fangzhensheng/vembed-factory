from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry


@LossRegistry.register("cosent")
class CoSENTLoss(nn.Module):
    """Cosine Sentence loss: log(1 + sum(exp(scale * (neg_sim - pos_sim))))

    More robust to hyperparameters than triplet loss and typically converges faster.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.scale = config.get("cosent_scale", 20.0)

    def _logsumexp_with_one(self, diffs: torch.Tensor) -> torch.Tensor:
        """Compute log(1 + sum(exp(scale * diffs))) via logsumexp([0, scale*diffs])."""
        scores = torch.cat([torch.zeros(1, device=diffs.device), self.scale * diffs])
        return torch.logsumexp(scores, dim=0)

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
    ):
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)
        pos_sim = (query_emb * positive_emb).sum(dim=1)

        if negative_emb is not None:
            negative_emb = F.normalize(negative_emb, p=2, dim=1)
            if negative_emb.dim() == 2:
                negative_emb = negative_emb.view(query_emb.size(0), -1, negative_emb.size(-1))
            neg_sim = torch.bmm(query_emb.unsqueeze(1), negative_emb.transpose(1, 2)).squeeze(1)
            diffs = (neg_sim - pos_sim.unsqueeze(1)).reshape(-1)
            return self._logsumexp_with_one(diffs)

        # In-batch negatives
        sim_matrix = query_emb @ positive_emb.T
        identity_mask = torch.eye(query_emb.size(0), device=query_emb.device, dtype=torch.bool)

        diffs = sim_matrix - pos_sim.unsqueeze(1)
        diffs.masked_fill_(identity_mask, -1e9)
        diffs = diffs.reshape(-1)
        diffs = diffs[diffs > -1e8]

        return self._logsumexp_with_one(diffs)
