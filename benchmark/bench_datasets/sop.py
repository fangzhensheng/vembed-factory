"""Stanford Online Products (SOP) benchmark — Image-to-Image retrieval."""

import json
import logging
import os
from typing import Any

import numpy as np
import torch

from benchmark.utils.mrl import evaluate_mrl

logger = logging.getLogger(__name__)

NAME = "sop"


def _find_first_file(root_dir: str, filename: str) -> str | None:
    if os.path.isfile(os.path.join(root_dir, filename)):
        return os.path.join(root_dir, filename)
    for current, _dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(current, filename)
    return None


def add_generate_arguments(parser):
    parser.add_argument("--sop_root", type=str, default="data/stanford_online_products")
    parser.add_argument("--sop_split", type=str, default="test", choices=["train", "test"])


def _load_sop_entries(sop_root: str, split: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    index_name = "Ebay_test.txt" if split == "test" else "Ebay_train.txt"
    index_path = _find_first_file(sop_root, index_name)
    if index_path is None:
        raise FileNotFoundError(f"Cannot find {index_name} under sop_root={sop_root}")

    base_dir = os.path.dirname(index_path)
    entries: list[dict[str, Any]] = []
    class_ids: list[int] = []

    with open(index_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower().startswith("image_id"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            _image_id, class_id, _super_class_id, rel_path = parts[0], parts[1], parts[2], parts[3]
            full_path = os.path.join(base_dir, rel_path)
            entries.append(
                {
                    "query": "",
                    "query_image": full_path,
                    "positive": full_path,
                    "class_id": class_id,
                }
            )
            class_ids.append(int(class_id))
    return entries, np.asarray(class_ids, dtype=np.int64)


def build_data(args) -> dict[str, Any]:
    entries, class_ids = _load_sop_entries(args.sop_root, args.sop_split)
    return {
        "data_source": entries,
        "image_root": "",
        "retrieval_mode": "i2i",
        "extras": {"class_ids": class_ids},
    }


def add_run_arguments(parser):
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_sop")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--sop_root", type=str, default="data/stanford_online_products")
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "colbert"],
        help="Similarity metric: 'cosine' for dense [N,D], 'colbert' (MaxSim) for token-level [N,L,D]",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        help="Override model pooling (e.g. 'none' for ColBERT)",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=None,
        help="Override projection dim (e.g. 128 for ColBERT)",
    )
    parser.add_argument(
        "--topk_tokens",
        type=int,
        default=None,
        help="Attention-guided top-K token pruning (0 = keep all tokens)",
    )
    parser.add_argument(
        "--mrl_dims",
        type=int,
        nargs="+",
        default=None,
        help="List of MRL dimensions to evaluate (e.g. 1536 1024 768)",
    )


def _compute_scores_cosine(
    q_t: torch.Tensor, d_t: torch.Tensor, start: int, end: int, device: torch.device
) -> torch.Tensor:
    """Compute dense cosine similarity: q [B, D] @ d.T [D, N]."""
    qb = q_t[start:end]
    return qb @ d_t.T


def _compute_scores_colbert(
    qb: torch.Tensor, d_flat_T: torch.Tensor, n_docs: int, l_d: int
) -> torch.Tensor:
    """Compute ColBERT MaxSim scores using brute-force matrix multiplication."""
    # [B, L_q, D] @ [D, N*L_d] → [B, L_q, N*L_d]
    sim = qb @ d_flat_T
    B, L_q = qb.shape[:2]
    # [B, L_q, N, L_d] → max over L_d → [B, L_q, N] → mean over L_q → [B, N]
    sim = sim.view(B, L_q, n_docs, l_d)
    return sim.max(dim=3).values.mean(dim=1).float()


def _sop_recall_at_k_from_embeddings(
    query_embeddings_path: str,
    doc_embeddings_path: str,
    class_ids_path: str,
    *,
    topk: int,
    output_dir: str,
    similarity: str = "cosine",
    mrl_dims: list[int] | None = None,
) -> dict[str, float]:
    q = np.load(query_embeddings_path).astype("float32")
    d = np.load(doc_embeddings_path).astype("float32")
    class_ids = np.load(class_ids_path).astype("int64")

    is_colbert = similarity == "colbert"

    if is_colbert:
        if q.ndim != 3 or d.ndim != 3:
            raise ValueError(
                f"ColBERT expects 3D arrays [N, L, D], got query={q.shape}, doc={d.shape}"
            )
        if q.shape[2] != d.shape[2]:
            raise ValueError(f"Dim mismatch: query D={q.shape[2]} doc D={d.shape[2]}")
    else:
        if q.ndim != 2 or d.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got query={q.shape}, doc={d.shape}")
        if q.shape[1] != d.shape[1]:
            raise ValueError(f"Dim mismatch: query={q.shape[1]} doc={d.shape[1]}")

    if q.shape[0] != d.shape[0] or q.shape[0] != class_ids.shape[0]:
        raise ValueError(
            f"Count mismatch: query={q.shape[0]} doc={d.shape[0]} class_ids={class_ids.shape[0]}"
        )

    unique_ids, counts = np.unique(class_ids, return_counts=True)
    count_per_sample = counts[np.searchsorted(unique_ids, class_ids)]
    valid = count_per_sample > 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cid_t = torch.from_numpy(class_ids).to(device)
    valid_t = torch.from_numpy(valid).to(device)

    n = q.shape[0]
    k = int(topk)
    k = min(k, n - 1) if n > 1 else 0

    cutoffs = [c for c in [1, 10, 100] if c <= k]
    if k not in cutoffs and k > 0:
        cutoffs.append(k)
    denom = int(valid.sum())

    # Pre-load data to GPU once if possible
    if is_colbert:
        q_full = torch.from_numpy(q).half().to(device)
        d_full = torch.from_numpy(d).half().to(device)
    else:
        q_full = torch.from_numpy(q).to(device)
        d_full = torch.from_numpy(d).to(device)

    # Standard Dense MRL
    if not is_colbert:

        def _dense_eval(q_slice: torch.Tensor, d_slice: torch.Tensor) -> dict[str, float]:
            totals = {c: 0 for c in cutoffs}
            batch_size = 256 if device.type == "cuda" else 64

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                scores = _compute_scores_cosine(q_slice, d_slice, start, end, device)

                row = torch.arange(end - start, device=device)
                col = torch.arange(start, end, device=device)
                scores[row, col] = torch.finfo(scores.dtype).min

                topk_idx = torch.topk(scores, k=k, dim=1).indices
                retrieved_cid = cid_t[topk_idx]
                query_cid = cid_t[start:end].unsqueeze(1)
                eq = retrieved_cid.eq(query_cid)

                vb = valid_t[start:end]
                for c in cutoffs:
                    hits = eq[:, :c].any(dim=1) & vb
                    totals[c] += int(hits.sum().item())
                del scores

            return {f"recall@{c}": (totals[c] / denom if denom > 0 else 0.0) for c in cutoffs}

        all_results = evaluate_mrl(
            q_full,
            d_full,
            mrl_dims,
            _dense_eval,
            full_dim=q.shape[1],
        )

    else:
        # ColBERT MRL Logic (Custom)
        if mrl_dims:
            logger.warning(
                "MRL evaluation for ColBERT is not fully supported via shared utility yet. Using full dim."
            )

        q_t = torch.nn.functional.normalize(q_full, p=2, dim=-1)
        d_t = torch.nn.functional.normalize(d_full, p=2, dim=-1)

        N_d, L_d, D = d_t.shape
        d_flat_T = d_t.reshape(N_d * L_d, D).T.contiguous()
        del d_t

        batch_size = 128
        totals = {c: 0 for c in cutoffs}

        logger.info(
            "ColBERT MaxSim (brute-force, fp16): N=%d, L=%d, D=%d, batch=%d, topk=%d",
            n,
            L_d,
            D,
            batch_size,
            k,
        )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            scores = _compute_scores_colbert(q_t[start:end], d_flat_T, N_d, L_d)

            row = torch.arange(end - start, device=device)
            col = torch.arange(start, end, device=device)
            scores[row, col] = torch.finfo(scores.dtype).min

            topk_idx = torch.topk(scores, k=k, dim=1).indices
            retrieved_cid = cid_t[topk_idx]
            query_cid = cid_t[start:end].unsqueeze(1)
            eq = retrieved_cid.eq(query_cid)

            vb = valid_t[start:end]
            for c in cutoffs:
                hits = eq[:, :c].any(dim=1) & vb
                totals[c] += int(hits.sum().item())
            del scores

        all_results = {f"recall@{c}": (totals[c] / denom if denom > 0 else 0.0) for c in cutoffs}

    report_name = "sop_colbert_report.json" if is_colbert else "sop_standard_report.json"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, report_name)
    with open(out_path, "w") as f:
        json.dump(
            {
                "similarity": similarity,
                "num_images": int(n),
                "num_valid_queries": int(denom),
                "topk": int(k),
                **all_results,
            },
            f,
            indent=2,
        )

    logger.info("Saved report to %s", out_path)
    return all_results


def run(args, *, run_cmd, script_dir):
    similarity = getattr(args, "similarity", "cosine")
    pooling = getattr(args, "pooling", None)
    projection_dim = getattr(args, "projection_dim", None)
    topk_tokens = getattr(args, "topk_tokens", None)

    gen_cmd = [
        "accelerate",
        "launch",
        os.path.join(script_dir, "generate_embeddings.py"),
        "--model_path",
        args.model_path,
        "--dataset",
        NAME,
        "--sop_root",
        args.sop_root,
        "--sop_split",
        "test",
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--subset",
        "test",
    ]
    if pooling:
        gen_cmd += ["--pooling", pooling]
    if projection_dim:
        gen_cmd += ["--projection_dim", str(projection_dim)]
    if topk_tokens:
        gen_cmd += ["--topk_tokens", str(topk_tokens)]
    run_cmd(gen_cmd)

    q_path = os.path.join(args.output_dir, "test_query_embeddings.npy")
    d_path = os.path.join(args.output_dir, "test_doc_embeddings.npy")
    cid_path = os.path.join(args.output_dir, "test_class_ids.npy")

    mrl_dims = getattr(args, "mrl_dims", None)

    _sop_recall_at_k_from_embeddings(
        q_path,
        d_path,
        cid_path,
        topk=args.topk,
        output_dir=args.output_dir,
        similarity=similarity,
        mrl_dims=mrl_dims,
    )
