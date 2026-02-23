"""
Step 2a: Dense retrieval evaluation without FAISS.

Encodes images/texts via VEmbedWrapper, computes cosine similarity,
and reports Recall@K + MRR.  Supports both dense (single-vector) and
late-interaction (multi-vector / MaxSim) models.

Usage:
    python benchmark/evaluate_dense.py \
        --model_path output_clip/checkpoint-best \
        --data_path data/flickr30k/test.jsonl \
        --image_root data/flickr30k
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import ImageListDataset, KarpathyDataset
from metrics import maxsim_matrix, mrr, recall_at_k
from model_wrapper import VEmbedWrapper


def _encode_images(
    wrapper: VEmbedWrapper, paths: list, device: str, batch_size: int
) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_paths = paths[i : i + batch_size]
        images = [ImageListDataset(batch_paths)[j] for j in range(len(batch_paths))]
        embs.append(wrapper.encode_image(images, device))
    return torch.cat(embs)


def _encode_texts(
    wrapper: VEmbedWrapper, texts: list, device: str, batch_size: int
) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        embs.append(wrapper.encode_text(texts[i : i + batch_size], device))
    return torch.cat(embs)


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model_path}")

    # Load
    wrapper = VEmbedWrapper(args.model_path)
    dataset = KarpathyDataset(args.data_path, args.image_root)

    print(f"Images: {dataset.num_images} unique, Captions: {dataset.num_captions}")

    # Encode
    img_embs = _encode_images(wrapper, dataset.unique_image_paths, device, args.batch_size)
    txt_embs = _encode_texts(wrapper, dataset.captions(), device, args.batch_size)

    # Score
    is_late_interaction = img_embs.dim() == 3
    if is_late_interaction:
        print("Using MaxSim (late interaction) scoring...")
        sim = maxsim_matrix(txt_embs, img_embs, device)
    else:
        sim = torch.matmul(txt_embs, img_embs.t())

    # Metrics
    gt = dataset.caption_to_image_idx
    recalls = recall_at_k(sim, gt, k_values=(1, 5, 10))
    mrr_score = mrr(sim, gt)

    results = {**recalls, "mrr": mrr_score}

    print()
    print("=" * 50)
    print(f"  Results: {args.model_path}")
    print("=" * 50)
    for name, val in results.items():
        print(f"  {name:<12} {val:.4f}")
    print("=" * 50)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dense_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
