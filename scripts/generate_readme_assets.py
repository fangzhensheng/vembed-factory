#!/usr/bin/env python3
"""
Generate retrieval visualization assets for the README.

Renders a grid showing query images alongside their Top-K retrieved results,
with green/red borders indicating correct/incorrect matches.

Usage:
    # SOP (image-to-image)
    python scripts/generate_readme_assets.py \
        --dataset sop \
        --emb_dir benchmark_output_sop_compare/after \
        --output docs/assets/sop_i2i_demo.png

    # Flickr30k (text-to-image)
    python scripts/generate_readme_assets.py \
        --dataset flickr30k \
        --emb_dir benchmark_output_flickr30k_compare/after \
        --output docs/assets/flickr30k_t2i_demo.png

    # Custom embedding dir + dataset root
    python scripts/generate_readme_assets.py \
        --dataset sop \
        --emb_dir my_output/embeddings \
        --data_root data/stanford_online_products \
        --num_queries 4 --topk 5 \
        --output docs/assets/custom_demo.png
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Colour palette
_GREEN = "#2ecc71"
_RED = "#e74c3c"
_BLUE = "#3498db"
_BG = "#fafafa"
_TEXT_DIM = "#888888"


# Dataset loaders


def _load_sop(data_root: str, emb_dir: str):
    """Return (query_embs, doc_embs, entries, class_ids, mode)."""
    from benchmark.bench_datasets.sop import _load_sop_entries

    entries, class_ids = _load_sop_entries(data_root, "test")
    q = np.load(os.path.join(emb_dir, "test_query_embeddings.npy"))
    d = np.load(os.path.join(emb_dir, "test_doc_embeddings.npy"))
    return q, d, entries, class_ids, "i2i"


def _load_flickr30k(data_root: str, emb_dir: str):
    """Return (query_embs, doc_embs, entries, image_ids, mode)."""
    import json

    jsonl_path = os.path.join(data_root, "test.jsonl")
    entries: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    q = np.load(os.path.join(emb_dir, "test_query_embeddings.npy"))
    d = np.load(os.path.join(emb_dir, "test_doc_embeddings.npy"))
    image_ids = np.load(os.path.join(emb_dir, "test_image_ids.npy"), allow_pickle=True)
    return q, d, entries, image_ids, "t2i"


_LOADERS = {
    "sop": (_load_sop, "data/stanford_online_products"),
    "flickr30k": (_load_flickr30k, "data/flickr30k"),
}


# Utilities


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _load_thumbnail(path: str, size: int = 224) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img
    except Exception:
        return Image.new("RGB", (size, size), (200, 200, 200))


def _pick_diverse_samples(
    entries: list[dict],
    class_ids: np.ndarray,
    q_norm: np.ndarray,
    d_norm: np.ndarray,
    num_queries: int,
    seed: int,
) -> list[int]:
    """Select query indices from distinct categories where Top-1 is correct."""
    rng = np.random.RandomState(seed)

    n = len(entries)
    scan = rng.choice(n, min(3000, n), replace=False)

    # Group by category — pick those with correct Top-1
    cat_map: dict[str, list[int]] = {}
    for i in scan:
        scores = q_norm[i] @ d_norm.T
        top1 = int(np.argmax(scores))
        if class_ids[i] == class_ids[top1] and top1 != i:
            cat = str(class_ids[i])
            cat_map.setdefault(cat, []).append(int(i))

    cats = list(cat_map.keys())
    if len(cats) >= num_queries:
        chosen = rng.choice(cats, num_queries, replace=False)
        return [int(rng.choice(cat_map[c])) for c in chosen]

    # Fallback
    return rng.choice(n, num_queries, replace=False).tolist()


def _pick_flickr_samples(
    entries: list[dict],
    image_ids: np.ndarray,
    q_norm: np.ndarray,
    d_norm: np.ndarray,
    num_queries: int,
    seed: int,
) -> list[int]:
    """Select caption indices that successfully retrieve the correct image in Top-1."""
    rng = np.random.RandomState(seed)

    # Deduplicate images
    seen: dict[str, int] = {}
    unique_embs: list[np.ndarray] = []
    cap_to_img: list[int] = []
    for i, img_id in enumerate(image_ids):
        img_id = str(img_id)
        if img_id not in seen:
            seen[img_id] = len(unique_embs)
            unique_embs.append(d_norm[i])
        cap_to_img.append(seen[img_id])
    d_unique = np.stack(unique_embs)

    n = len(entries)
    scan = rng.choice(n, min(2000, n), replace=False)
    good: list[int] = []

    for i in scan:
        scores = q_norm[i] @ d_unique.T
        top1 = int(np.argmax(scores))
        if top1 == cap_to_img[i]:
            good.append(int(i))
        if len(good) >= num_queries * 3:
            break

    if len(good) >= num_queries:
        return rng.choice(good, num_queries, replace=False).tolist()
    return rng.choice(n, num_queries, replace=False).tolist()


# Rendering


def _render_sop(
    indices: list[int],
    entries: list[dict],
    class_ids: np.ndarray,
    q_norm: np.ndarray,
    d_norm: np.ndarray,
    topk: int,
    output_path: str,
):
    """Render SOP i2i retrieval grid."""
    n_rows = len(indices)
    n_cols = 1 + topk  # Query + Top-K
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.8 * n_rows),
        facecolor=_BG,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "SOP Image-to-Image Retrieval — Top-5",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for row, idx in enumerate(indices):
        scores = q_norm[idx] @ d_norm.T
        # Exclude self
        scores[idx] = -np.inf
        top_ids = np.argsort(scores)[::-1][:topk]

        # Query column
        ax = axes[row, 0]
        img = _load_thumbnail(entries[idx]["query_image"])
        ax.imshow(img)
        ax.set_title("Query", fontsize=10, fontweight="bold", color=_BLUE)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BLUE)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

        # Retrieved columns
        for col, rid in enumerate(top_ids, start=1):
            ax = axes[row, col]
            img = _load_thumbnail(entries[rid]["positive"])
            ax.imshow(img)

            match = class_ids[idx] == class_ids[rid]
            colour = _GREEN if match else _RED
            label = "\u2713" if match else "\u2717"

            ax.set_title(
                f"#{col}  {scores[rid]:.3f}  {label}",
                fontsize=9,
                fontweight="bold",
                color=colour,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _render_flickr30k(
    indices: list[int],
    entries: list[dict],
    image_ids: np.ndarray,
    q_norm: np.ndarray,
    d_norm: np.ndarray,
    data_root: str,
    topk: int,
    output_path: str,
):
    """Render Flickr30k t2i retrieval grid (text query → top-K images)."""
    # Deduplicate images
    seen: dict[str, int] = {}
    unique_paths: list[str] = []
    unique_embs: list[np.ndarray] = []
    cap_to_img: list[int] = []
    for i, img_id in enumerate(image_ids):
        img_id = str(img_id)
        if img_id not in seen:
            seen[img_id] = len(unique_embs)
            unique_paths.append(os.path.join(data_root, entries[i]["positive"]))
            unique_embs.append(d_norm[i])
        cap_to_img.append(seen[img_id])
    d_unique = np.stack(unique_embs)

    n_rows = len(indices)
    n_cols = topk
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 3.4 * n_rows),
        facecolor=_BG,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Flickr30k Text-to-Image Retrieval — Top-5",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for row, idx in enumerate(indices):
        caption = entries[idx]["query"]
        gt_img_idx = cap_to_img[idx]

        scores = q_norm[idx] @ d_unique.T
        top_ids = np.argsort(scores)[::-1][:topk]

        # Show caption as a row label on the left margin
        wrapped = textwrap.fill(caption, width=40)
        fig.text(
            0.01,
            1.0 - (row + 0.5) / n_rows,
            f"\u201c{wrapped}\u201d",
            fontsize=8,
            fontstyle="italic",
            color=_TEXT_DIM,
            va="center",
            ha="left",
            transform=fig.transFigure,
        )

        for col, rid in enumerate(top_ids):
            ax = axes[row, col]
            img = _load_thumbnail(unique_paths[rid])
            ax.imshow(img)

            match = rid == gt_img_idx
            colour = _GREEN if match else _RED
            label = "\u2713" if match else "\u2717"

            ax.set_title(
                f"#{col + 1}  {scores[rid]:.3f}  {label}",
                fontsize=9,
                fontweight="bold",
                color=colour,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0.18, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# CLI

_DEFAULT_OUTPUTS = {
    "sop": "docs/assets/sop_i2i_demo.png",
    "flickr30k": "docs/assets/flickr30k_t2i_demo.png",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval demo images for the README.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(_LOADERS.keys()),
        help="Dataset to visualize",
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Directory containing *_query_embeddings.npy / *_doc_embeddings.npy",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (auto-detected)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of query rows")
    parser.add_argument("--topk", type=int, default=5, help="Number of retrieved results per query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    loader_fn, default_root = _LOADERS[args.dataset]
    data_root = args.data_root or default_root
    output_path = args.output or _DEFAULT_OUTPUTS[args.dataset]

    print(f"  Dataset  : {args.dataset}")
    print(f"  Emb dir  : {args.emb_dir}")
    print(f"  Data root: {data_root}")
    print(f"  Output   : {output_path}")
    print()

    # Load
    q, d, entries, meta, mode = loader_fn(data_root, args.emb_dir)

    # Align lengths
    min_len = min(len(entries), len(q), len(d))
    q, d, entries = q[:min_len], d[:min_len], entries[:min_len]
    if hasattr(meta, "__len__") and len(meta) > min_len:
        meta = meta[:min_len]

    q_norm = _normalise(q)
    d_norm = _normalise(d)

    # Pick samples
    if args.dataset == "sop":
        indices = _pick_diverse_samples(
            entries,
            meta,
            q_norm,
            d_norm,
            args.num_queries,
            args.seed,
        )
        _render_sop(indices, entries, meta, q_norm, d_norm, args.topk, output_path)

    elif args.dataset == "flickr30k":
        indices = _pick_flickr_samples(
            entries,
            meta,
            q_norm,
            d_norm,
            args.num_queries,
            args.seed,
        )
        _render_flickr30k(
            indices,
            entries,
            meta,
            q_norm,
            d_norm,
            data_root,
            args.topk,
            output_path,
        )

    print("  Done!")


if __name__ == "__main__":
    main()
