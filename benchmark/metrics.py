"""
Retrieval metrics: Recall@K, MRR, and MaxSim scoring for late-interaction models.
"""

import numpy as np
import torch


def recall_at_k(
    sim_matrix: torch.Tensor, gt_indices: list[int], k_values=(1, 5, 10)
) -> dict[str, float]:
    """Compute Recall@K from a (num_queries x num_docs) similarity matrix."""
    max_k = max(k_values)
    _, top_indices = torch.topk(sim_matrix, k=min(max_k, sim_matrix.size(1)), dim=1)
    top_indices = top_indices.numpy()

    gt = np.array(gt_indices)
    results = {}
    for k in k_values:
        hits = np.any(top_indices[:, :k] == gt[:, None], axis=1)
        results[f"recall@{k}"] = float(hits.mean())
    return results


def mrr(sim_matrix: torch.Tensor, gt_indices: list[int]) -> float:
    """Mean Reciprocal Rank."""
    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    sorted_indices = sorted_indices.numpy()
    gt = np.array(gt_indices)

    # Find rank of ground-truth for each query
    ranks = (sorted_indices == gt[:, None]).argmax(axis=1) + 1
    return float((1.0 / ranks).mean())


def maxsim_matrix(
    txt_embs: torch.Tensor,
    img_embs: torch.Tensor,
    device: str = "cpu",
    doc_batch_size: int = 16,
) -> torch.Tensor:
    """
    Late-interaction (ColBERT-style) scoring.
    txt_embs: (N_text, L_q, D)
    img_embs: (N_img, L_d, D)
    Returns: (N_text, N_img) similarity matrix.
    """
    sim = torch.zeros(len(txt_embs), len(img_embs))

    for i in range(len(txt_embs)):
        q = txt_embs[i].to(device)  # (L_q, D)
        scores_row = []
        for j in range(0, len(img_embs), doc_batch_size):
            d_batch = img_embs[j : j + doc_batch_size].to(device)  # (B, L_d, D)
            # (L_q, D) x (B, L_d, D) -> (B, L_q, L_d)
            token_sim = torch.einsum("qd,bld->bql", q, d_batch)
            # MaxSim: max over doc tokens, then sum over query tokens
            scores_row.append(token_sim.max(dim=2).values.sum(dim=1).cpu())
        sim[i] = torch.cat(scores_row)

    return sim
