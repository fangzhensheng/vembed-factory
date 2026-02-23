from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry


def _maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """Compute MaxSim scores between queries and documents.

    Token embeddings are L2-normalised inside this function so that each
    per-token dot product lies in ``[-1, 1]``.  The final score is the
    **mean** (not sum) of per-query-token maxima, making it robust to
    variable sequence lengths (e.g. 257 image patches vs 32 text tokens).

    Args:
        query_emb: ``[B_q, L_q, D]`` — query token embeddings.
        doc_emb:   ``[B_d, L_d, D]`` — document token embeddings.

    Returns:
        ``[B_q, B_d]`` score matrix.
    """
    # L2-normalise each token vector so dot products are cosine similarities
    query_emb = F.normalize(query_emb, p=2, dim=-1)
    doc_emb = F.normalize(doc_emb, p=2, dim=-1)

    # token-level similarity: (B_q, B_d, L_q, L_d)
    token_sim = torch.einsum("bqd,ckd->bcqk", query_emb, doc_emb)
    # MaxSim: max over doc tokens, then **mean** over query tokens → (B_q, B_d)
    return token_sim.max(dim=3).values.mean(dim=2)


@LossRegistry.register("colbert")
class ColBERTLoss(nn.Module):
    """Late-interaction (ColBERT) loss with in-batch + optional hard negatives.

    ``Score(Q, D) = mean_q max_d (q_i · d_j)``

    Supports **label-aware false-negative masking**: if ``labels`` is provided,
    same-class samples in the batch are excluded from the negative set (their
    logits are set to ``-inf`` before cross-entropy), preventing the model from
    being penalised for ranking same-class items highly.

    When ``negative_emb`` is provided the score matrix is expanded to include
    hard negatives so the model is trained to rank positives above both
    in-batch and hard negatives.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.temperature = config.get("temperature", 0.05)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        B = query_emb.size(0)

        if negative_emb is None:
            # In-batch negatives only: score matrix [B, B]
            scores = _maxsim(query_emb, positive_emb) / self.temperature

            # In SOP-like datasets multiple images share the same class_id.
            # If sample j has the same label as sample i (j ≠ i) it is a
            # false negative — mask it out so the model isn't penalised for
            # ranking it highly.
            if labels is not None:
                label_col = labels.view(-1, 1)  # [B, 1]
                same_label = label_col.eq(label_col.T)  # [B, B]
                eye = torch.eye(B, dtype=torch.bool, device=scores.device)
                false_neg_mask = same_label & ~eye  # same class, different index
                scores = scores.masked_fill(false_neg_mask, -1e9)

            target = torch.arange(B, device=query_emb.device)
            return self.cross_entropy(scores, target)

        all_docs = torch.cat([positive_emb, negative_emb], dim=0)
        scores = _maxsim(query_emb, all_docs) / self.temperature

        # False-negative masking for the in-batch portion (first B columns)
        if labels is not None:
            label_col = labels.view(-1, 1)
            same_label = label_col.eq(label_col.T)
            eye = torch.eye(B, dtype=torch.bool, device=scores.device)
            false_neg_mask = same_label & ~eye
            scores[:, :B] = scores[:, :B].masked_fill(false_neg_mask, -1e9)

        target = torch.arange(B, device=query_emb.device)
        return self.cross_entropy(scores, target)
