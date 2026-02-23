import numpy as np


def compute_metrics(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    labels: list[int] | None = None,
    top_k: list[int] | None = None,
) -> dict[str, float]:
    if top_k is None:
        top_k = [1, 5, 10]
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    sim_matrix = query_embeddings @ doc_embeddings.T
    num_queries = sim_matrix.shape[0]

    gt_indices = np.arange(num_queries) if labels is None else np.asarray(labels)

    max_k = max(top_k)
    # Partial argsort: only the top-max_k indices matter
    top_indices = np.argsort(sim_matrix, axis=1)[:, -max_k:][:, ::-1]

    metrics: dict[str, float] = {}
    for k in top_k:
        hits = np.any(top_indices[:, :k] == gt_indices[:, None], axis=1)
        metrics[f"Recall@{k}"] = hits.mean()

    # MRR: reciprocal rank of the first correct hit within top-max_k
    match_positions = top_indices == gt_indices[:, None]
    # For each query, find the column of the first True (or -1 if absent)
    has_match = match_positions.any(axis=1)
    first_rank = match_positions.argmax(axis=1) + 1  # 1-indexed
    reciprocal_ranks = np.where(has_match, 1.0 / first_rank, 0.0)
    metrics["MRR"] = reciprocal_ranks.mean()

    return metrics
