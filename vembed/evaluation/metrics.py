import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union


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


def compute_recall_at_k(
    query_embeddings: Union[torch.Tensor, np.ndarray],
    doc_embeddings: Union[torch.Tensor, np.ndarray],
    query_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    doc_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    k_list: list[int] = [1, 10, 100],
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """Compute Recall@k metrics for label-based retrieval evaluation.

    This function computes recall metrics where relevance is determined by label matching,
    not by index matching. This is suitable for scenarios where multiple items share the
    same label (e.g., multi-label retrieval, clustering evaluation).

    Args:
        query_embeddings: Query embeddings of shape [num_queries, embedding_dim]
                         Can be torch.Tensor or np.ndarray
        doc_embeddings: Document embeddings of shape [num_docs, embedding_dim]
                       Can be torch.Tensor or np.ndarray
        query_labels: Query labels of shape [num_queries]. If None, returns empty dict.
        doc_labels: Document labels of shape [num_docs]. If None, uses query_labels.
        k_list: List of k values to compute recall at (e.g., [1, 10, 100])
        exclude_diagonal: If True, excludes diagonal (self-matching) from top-k
                         when query_embeddings and doc_embeddings are the same set.

    Returns:
        Dictionary with keys like "recall@1", "recall@10", "recall@100"
        Example: {"recall@1": 0.95, "recall@10": 0.98, "recall@100": 0.99}

    Notes:
        - If labels are not provided, returns empty dict
        - Computes cosine similarity between queries and documents
        - Excludes diagonal (self-matching) from top-k when appropriate
        - Only considers pairs with matching labels as relevant
        - Handles in-batch retrieval scenarios automatically
    """
    if query_labels is None:
        return {}

    # Convert to torch tensors if needed
    if isinstance(query_embeddings, np.ndarray):
        query_embeddings = torch.from_numpy(query_embeddings)
    if isinstance(doc_embeddings, np.ndarray):
        doc_embeddings = torch.from_numpy(doc_embeddings)
    if isinstance(query_labels, np.ndarray):
        query_labels = torch.from_numpy(query_labels)
    if isinstance(doc_labels, np.ndarray):
        doc_labels = torch.from_numpy(doc_labels)

    if doc_labels is None:
        doc_labels = query_labels

    # Ensure tensors are on CPU for computation
    query_embeddings = query_embeddings.cpu().float()
    doc_embeddings = doc_embeddings.cpu().float()
    query_labels = query_labels.cpu().long()
    doc_labels = doc_labels.cpu().long()

    num_queries = query_embeddings.shape[0]
    num_docs = doc_embeddings.shape[0]

    # L2 normalize embeddings for cosine similarity
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    # Compute similarity matrix: [num_queries, num_docs]
    similarity = torch.mm(query_embeddings, doc_embeddings.T)

    # Create label matching matrix: [num_queries, num_docs]
    query_labels_expanded = query_labels.view(-1, 1)
    label_match = query_labels_expanded.eq(doc_labels.view(1, -1)).float()

    # Exclude diagonal (self-matching) for in-batch scenarios
    if exclude_diagonal and num_queries == num_docs:
        mask = torch.eye(num_queries, num_docs, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float('-inf'))

    # Get max k from k_list for efficiency
    max_k = max(k_list) if k_list else 1
    max_k = min(max_k, num_docs)

    # Get top-k indices: [num_queries, max_k]
    _, topk_indices = torch.topk(similarity, k=max_k, dim=1)

    # Check if top-k items have matching labels
    topk_labels_match = label_match.gather(1, topk_indices)

    # Compute recall@k: for each query, check if any of top-k docs match the label
    metrics = {}
    for k in k_list:
        if k > num_docs:
            actual_k = num_docs
        else:
            actual_k = k

        # Recall@k: fraction of queries that have at least one match in top-k
        topk_match = topk_labels_match[:, :actual_k]
        recall_k = (topk_match.sum(dim=1) > 0).float().mean().item()

        metrics[f"recall@{k}"] = recall_k

    return metrics


def compute_recall_metrics(
    all_query_embeddings: list[Union[torch.Tensor, np.ndarray]],
    all_doc_embeddings: list[Union[torch.Tensor, np.ndarray]],
    all_query_labels: Optional[list[Union[torch.Tensor, np.ndarray]]] = None,
    all_doc_labels: Optional[list[Union[torch.Tensor, np.ndarray]]] = None,
    k_list: list[int] = [1, 10, 100],
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """Compute label-based recall metrics from batched embeddings and labels.

    Useful for eval_phase where embeddings and labels are accumulated in batches
    across multiple validation steps and processes (in distributed training).

    Args:
        all_query_embeddings: List of query embedding batches
        all_doc_embeddings: List of document embedding batches
        all_query_labels: List of query label batches (optional, required for recall)
        all_doc_labels: List of document label batches (optional)
        k_list: List of k values for recall computation
        exclude_diagonal: If True, excludes diagonal from top-k when appropriate

    Returns:
        Dictionary with recall metrics (empty dict if no labels provided)

    Example:
        >>> # In eval loop
        >>> all_q_embs, all_p_embs, all_q_labels = [], [], []
        >>> for batch in val_dataloader:
        ...     q_embs = model(...)
        ...     p_embs = model(...)
        ...     all_q_embs.append(accelerator.gather_for_metrics(q_embs).cpu())
        ...     all_p_embs.append(accelerator.gather_for_metrics(p_embs).cpu())
        ...     if "labels" in batch:
        ...         all_q_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())
        >>> metrics = compute_recall_metrics(all_q_embs, all_p_embs, all_q_labels)
    """
    if not all_query_embeddings or not all_doc_embeddings:
        return {}

    # Concatenate all batches
    query_embeddings = torch.cat(all_query_embeddings, dim=0)
    doc_embeddings = torch.cat(all_doc_embeddings, dim=0)

    query_labels = None
    if all_query_labels:
        query_labels = torch.cat(all_query_labels, dim=0)

    doc_labels = None
    if all_doc_labels:
        doc_labels = torch.cat(all_doc_labels, dim=0)

    return compute_recall_at_k(
        query_embeddings,
        doc_embeddings,
        query_labels,
        doc_labels,
        k_list,
        exclude_diagonal,
    )
