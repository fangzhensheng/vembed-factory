"""
Dense retrieval evaluation without FAISS.

Encodes images/texts via VEmbedWrapper, computes cosine similarity,
and reports Recall@K + MRR. Supports both dense (single-vector) and
late-interaction (multi-vector / MaxSim) models.

Usage:
    # Multimodal models (CLIP, SigLIP, Qwen3-VL)
    python benchmark/evaluate_dense.py \\
        --model_path output_clip/checkpoint-best \\
        --data_path data/flickr30k/test.jsonl \\
        --image_root data/flickr30k

    # Text-only models (Qwen3-Embedding, BGE)
    python benchmark/evaluate_dense.py \\
        --model_path experiments/output_qwen3_embedding \\
        --encoder_mode qwen3_embedding \\
        --data_path data/beir/test.jsonl
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
from model_wrapper import VEmbedWrapper, _auto_detect_encoder_mode


def _encode_images(
    wrapper: VEmbedWrapper, paths: list, device: str, batch_size: int
) -> torch.Tensor:
    if not wrapper.supports_images:
        raise RuntimeError(
            "Model does not support image encoding. "
            "This is a text-only model (e.g., qwen3_embedding)."
        )
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
    parser = argparse.ArgumentParser(
        description="Dense retrieval evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoder_mode", type=str, default=None,
                       help="Encoder mode (auto-detect if not specified). "
                            "Options: auto, qwen3_vl, qwen3_embedding, siglip, composed, etc.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None,
                       help="Image root directory (omit for text-only models)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model_path}")

    # Auto-detect encoder_mode if not specified
    encoder_mode = args.encoder_mode
    if encoder_mode is None:
        encoder_mode = _auto_detect_encoder_mode(args.model_path)
        if encoder_mode:
            print(f"Auto-detected encoder_mode: {encoder_mode}")
        else:
            print("Using encoder_mode: auto")

    # Load model
    wrapper = VEmbedWrapper(args.model_path, encoder_mode=encoder_mode)

    # Check if model supports images
    has_images = wrapper.supports_images
    print(f"Model supports images: {has_images}")

    if has_images and not args.image_root:
        parser.error("--image_root is required for models that support images")

    if not has_images and args.image_root:
        print("Warning: Model is text-only, ignoring --image_root")

    # Load dataset
    if has_images:
        dataset = KarpathyDataset(args.data_path, args.image_root)
        print(f"Images: {dataset.num_images} unique, Captions: {dataset.num_captions}")
    else:
        # Text-only mode - load simple JSONL dataset
        from benchmark.dataset import JsonlDataset
        dataset = JsonlDataset(args.data_path)
        print(f"Queries: {len(dataset.queries)}, Documents: {len(dataset.documents)}")

    # Encode
    if has_images:
        img_embs = _encode_images(wrapper, dataset.unique_image_paths, device, args.batch_size)
    txt_embs = _encode_texts(wrapper, dataset.captions() if has_images else dataset.queries, device, args.batch_size)

    # Score
    is_late_interaction = txt_embs.dim() == 3 or (has_images and img_embs.dim() == 3)
    if is_late_interaction:
        print("Using MaxSim (late interaction) scoring...")
        sim = maxsim_matrix(txt_embs, img_embs if has_images else txt_embs, device)
    else:
        if has_images:
            sim = torch.matmul(txt_embs, img_embs.t())
        else:
            # Text-to-text: compute similarity matrix
            sim = torch.matmul(txt_embs, txt_embs.t())

    # Metrics
    if has_images:
        gt = dataset.caption_to_image_idx
        recalls = recall_at_k(sim, gt, k_values=(1, 5, 10))
        mrr_score = mrr(sim, gt)
        results = {**recalls, "mrr": mrr_score}
    else:
        # Text-only metrics (use BEIR-style evaluation)
        from benchmark.metrics import recall_at_k as recall_k
        # For text-to-text, we assume diagonal or use provided relevance
        # Simplified: compute R@1, R@5, R@10 assuming first query matches first doc, etc.
        # In practice, you'd want proper relevance judgments
        results = {"text_only_mode": True, "note": "Use BEIR evaluation for proper metrics"}

    print()
    print("=" * 50)
    print(f"  Results: {args.model_path}")
    print("=" * 50)
    for name, val in results.items():
        if isinstance(val, float):
            print(f"  {name:<12} {val:.4f}")
        else:
            print(f"  {name:<12} {val}")
    print("=" * 50)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dense_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
