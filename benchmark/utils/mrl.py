"""Shared utilities for Matryoshka Representation Learning (MRL) evaluation."""

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def evaluate_mrl(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    mrl_dims: list[int] | None,
    eval_fn: Callable[[torch.Tensor, torch.Tensor], dict[str, float]],
    full_dim: int | None = None,
) -> dict[str, float]:
    """Evaluate performance across multiple MRL dimensions by slicing and normalizing embeddings."""
    if full_dim is None:
        full_dim = query_embeddings.shape[1]

    if not mrl_dims:
        mrl_dims = [full_dim]

    all_results = {}

    for dim in mrl_dims:
        if dim > full_dim:
            logger.warning("Requested MRL dim %d > feature dim %d, skipping.", dim, full_dim)
            continue

        logger.info("Evaluating MRL Dimension: %d", dim)

        # Slice and Normalize
        # Ensure we are working with normalized vectors for each slice
        q_slice = F.normalize(query_embeddings[:, :dim], p=2, dim=1)
        d_slice = F.normalize(doc_embeddings[:, :dim], p=2, dim=1)

        # Run the dataset-specific evaluation logic
        dim_results = eval_fn(q_slice, d_slice)

        # Store results
        if len(mrl_dims) > 1:
            for k, v in dim_results.items():
                all_results[f"{k}_d{dim}"] = v
        else:
            all_results.update(dim_results)

        # Print summary for this dimension
        _log_dim_summary(dim_results, dim)

    return all_results


def _log_dim_summary(results: dict[str, float], dim: int) -> None:
    """Log metrics for a specific dimension in a tabular format."""
    header = f"  {'Metric (d=' + str(dim) + ')':<25} {'Value':>10}"
    divider = f"  {'─' * 37}"

    logger.info(header)
    logger.info(divider)

    # Sort keys for consistent output
    keys = sorted(results.keys())

    # Group by common prefixes if possible (t2i, i2t)
    t2i_keys = [k for k in keys if k.startswith("t2i")]
    i2t_keys = [k for k in keys if k.startswith("i2t")]
    other_keys = [k for k in keys if k not in t2i_keys and k not in i2t_keys]

    for k in t2i_keys:
        logger.info(f"    {k:<21} {results[k]:10.4f}")

    if i2t_keys:
        logger.info(divider)
        for k in i2t_keys:
            logger.info(f"    {k:<21} {results[k]:10.4f}")

    if other_keys:
        logger.info(divider)
        for k in other_keys:
            logger.info(f"    {k:<21} {results[k]:10.4f}")
    logger.info(divider)
