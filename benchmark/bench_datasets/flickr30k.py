"""Flickr30k benchmark — Karpathy split retrieval (t2i + i2t)."""

import json
import logging
import os
from typing import Any

import numpy as np
import torch

from benchmark.utils.mrl import evaluate_mrl

logger = logging.getLogger(__name__)

NAME = "flickr30k"
RUN_HELP = "Flickr30k Karpathy-split retrieval benchmark (t2i + i2t)"

_DEFAULT_ROOT = "data/flickr30k"
_DEFAULT_SPLIT = "test"
_REPORT_NAME = "flickr30k_report.json"


def add_generate_arguments(parser):
    parser.add_argument("--flickr_root", type=str, default=_DEFAULT_ROOT)
    parser.add_argument(
        "--flickr_split",
        type=str,
        default=_DEFAULT_SPLIT,
        choices=["train", "val", "test"],
    )


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def build_data(args) -> dict[str, Any]:
    split = getattr(args, "flickr_split", _DEFAULT_SPLIT)
    root = getattr(args, "flickr_root", _DEFAULT_ROOT)
    jsonl_path = os.path.join(root, f"{split}.jsonl")

    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"Flickr30k {split} data not found at {jsonl_path}\n"
            "  Run: python examples/prepare_data.py flickr30k"
        )

    entries = _load_jsonl(jsonl_path)
    image_ids = [e.get("image_id", os.path.basename(e["positive"])) for e in entries]

    return {
        "data_source": jsonl_path,
        "image_root": root,
        "retrieval_mode": "t2i",
        "extras": {"image_ids": np.array(image_ids, dtype=object)},
    }


def add_run_arguments(parser):
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_flickr30k")
    parser.add_argument("--flickr_root", type=str, default=_DEFAULT_ROOT)
    parser.add_argument(
        "--encoder_mode",
        type=str,
        default="auto",
        help="Encoder mode (e.g. qwen3_vl). Determines collator & processor.",
    )
    parser.add_argument(
        "--flickr_split",
        type=str,
        default=_DEFAULT_SPLIT,
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--mrl_dims",
        type=int,
        nargs="+",
        default=None,
        help="List of MRL dimensions to evaluate (e.g. 1536 1024 768)",
    )


def _evaluate_flickr30k(
    query_emb_path: str,
    doc_emb_path: str,
    image_ids_path: str,
    *,
    output_dir: str,
    mrl_dims: list[int] = None,
) -> dict[str, float]:
    """Evaluate performance on Flickr30k with image deduplication."""
    q_all = np.load(query_emb_path).astype("float32")
    d_all = np.load(doc_emb_path).astype("float32")
    image_ids = np.load(image_ids_path, allow_pickle=True)

    n_captions = q_all.shape[0]

    # Deduplicate images based on image_ids
    seen: dict[str, int] = {}
    caption_to_img_idx: list[int] = []
    unique_img_embs: list[np.ndarray] = []

    for i, img_id in enumerate(image_ids):
        img_id = str(img_id)
        if img_id not in seen:
            seen[img_id] = len(unique_img_embs)
            unique_img_embs.append(d_all[i])
        caption_to_img_idx.append(seen[img_id])

    unique_img_embs = np.stack(unique_img_embs)
    caption_to_img_idx = np.array(caption_to_img_idx)
    n_images = unique_img_embs.shape[0]

    img_to_caption_idxs: dict[int, list[int]] = {}
    for cap_i, img_i in enumerate(caption_to_img_idx):
        img_to_caption_idxs.setdefault(int(img_i), []).append(cap_i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_t_full = torch.from_numpy(q_all).to(device)
    d_t_full = torch.from_numpy(unique_img_embs).to(device)
    gt_t2i = torch.from_numpy(caption_to_img_idx).to(device)

    def _eval_fn(q_t: torch.Tensor, d_t: torch.Tensor) -> dict[str, float]:
        k_values = (1, 5, 10)
        results = {}

        # Text-to-Image
        sim_t2i = q_t @ d_t.T
        for k in k_values:
            topk_idx = torch.topk(sim_t2i, k=min(k, n_images), dim=1).indices
            hits = (topk_idx == gt_t2i.unsqueeze(1)).any(dim=1)
            results[f"t2i_recall@{k}"] = float(hits.float().mean().item())

        ranks_t2i = torch.argsort(sim_t2i, dim=1, descending=True) == gt_t2i.unsqueeze(1)
        ranks_t2i = ranks_t2i.float().argmax(dim=1) + 1
        results["t2i_mrr"] = float((1.0 / ranks_t2i.float()).mean().item())

        # Image-to-Text
        sim_i2t = d_t @ q_t.T
        for k in k_values:
            topk_idx = torch.topk(sim_i2t, k=min(k, n_captions), dim=1).indices
            hits = 0
            for img_i in range(n_images):
                gt_caps = set(img_to_caption_idxs[img_i])
                retrieved = topk_idx[img_i].cpu().tolist()
                if any(r in gt_caps for r in retrieved):
                    hits += 1
            results[f"i2t_recall@{k}"] = hits / n_images

        sorted_i2t = torch.argsort(sim_i2t, dim=1, descending=True).cpu().numpy()
        mrr_sum = 0.0
        for img_i in range(n_images):
            gt_caps = set(img_to_caption_idxs[img_i])
            for rank, idx in enumerate(sorted_i2t[img_i], start=1):
                if idx in gt_caps:
                    mrr_sum += 1.0 / rank
                    break
        results["i2t_mrr"] = mrr_sum / n_images

        results["rsum"] = sum(results[f"{d}_recall@{k}"] for d in ("t2i", "i2t") for k in k_values)
        return results

    all_results = evaluate_mrl(
        q_t_full,
        d_t_full,
        mrl_dims,
        _eval_fn,
        full_dim=q_all.shape[1],
    )

    report = {
        "num_images": int(n_images),
        "num_captions": int(n_captions),
        **{k: round(v, 6) for k, v in all_results.items()},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, _REPORT_NAME)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved report to %s", out_path)
    return all_results


def run(args, *, run_cmd, script_dir):
    import sys

    split = getattr(args, "flickr_split", _DEFAULT_SPLIT)
    encoder_mode = getattr(args, "encoder_mode", "auto")

    # If mrl_dims are specified, ensure we don't pass them to generate_embeddings.py
    # because it doesn't support them (it just generates full embeddings).
    # We handle slicing in _evaluate_flickr30k.

    gen_cmd = [
        sys.executable,
        os.path.join(script_dir, "generate_embeddings.py"),
        "--model_path",
        args.model_path,
        "--dataset",
        NAME,
        "--flickr_root",
        args.flickr_root,
        "--flickr_split",
        split,
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--encoder_mode",
        encoder_mode,
        "--subset",
        split,
    ]
    run_cmd(gen_cmd)

    prefix = split
    mrl_dims = args.mrl_dims
    _evaluate_flickr30k(
        query_emb_path=os.path.join(args.output_dir, f"{prefix}_query_embeddings.npy"),
        doc_emb_path=os.path.join(args.output_dir, f"{prefix}_doc_embeddings.npy"),
        image_ids_path=os.path.join(args.output_dir, f"{prefix}_image_ids.npy"),
        output_dir=args.output_dir,
        mrl_dims=mrl_dims,
    )
