"""
Encode queries and documents into .npy embedding files.

Supports distributed execution via Accelerate:
    accelerate launch benchmark/generate_embeddings.py --model_path ... --data_path ...

Output:
    {output_dir}/{subset}_query_embeddings.npy  — (N_queries, D)
    {output_dir}/{subset}_doc_embeddings.npy    — (N_docs, D)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vembed.data  # noqa: F401
from benchmark.bench_datasets.registry import discover_dataset_modules
from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry


def _split_query_pos(batch: dict, retrieval_mode: str):
    """Extract query and positive sub-batches from a collated eval batch.

    VLMs like Qwen3-VL represent images as placeholder tokens inside a text
    sequence, so image items carry ``input_ids`` / ``attention_mask`` as well
    as ``pixel_values`` / ``image_grid_thw``.
    """
    if retrieval_mode.startswith("i"):
        q_batch: dict = {"pixel_values": batch["query_pixel_values"]}
        # VLM: images also need input_ids
        if "query_input_ids" in batch:
            q_batch["input_ids"] = batch["query_input_ids"]
            q_batch["attention_mask"] = batch["query_attention_mask"]
        if "query_image_grid_thw" in batch and batch["query_image_grid_thw"] is not None:
            q_batch["image_grid_thw"] = batch["query_image_grid_thw"]
    else:
        q_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

    if retrieval_mode.endswith("t"):
        p_batch: dict = {
            "input_ids": batch["pos_input_ids"],
            "attention_mask": batch["pos_attention_mask"],
        }
    else:
        # Prefer prefixed key, fallback to legacy alias
        pv = batch.get("pos_pixel_values")
        if pv is None:
            pv = batch.get("pixel_values")
        p_batch = {"pixel_values": pv}

        # VLM: image items also need input_ids
        if "pos_input_ids" in batch:
            p_batch["input_ids"] = batch["pos_input_ids"]
            p_batch["attention_mask"] = batch["pos_attention_mask"]

        # Ensure we don't accidentally pick up query inputs if pos keys are missing
        # But for Qwen, the collator puts image input_ids in pos_input_ids

        # image_grid_thw
        grid = batch.get("pos_image_grid_thw")
        if grid is None:
            # Only fallback if retrieval mode suggests we might be looking at mixed batch
            # But usually it's safer to stick to prefixed if available
            grid = batch.get("image_grid_thw")

        if grid is not None:
            p_batch["image_grid_thw"] = grid

    return q_batch, p_batch


def _encode_batch(model, feature_model, batch):
    """Helper to encode a batch using the model."""
    if not batch:
        return None, "none"

    device = next(model.parameters()).device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    return model(**batch), "model"


def _maybe_first(embs):
    if isinstance(embs, torch.Tensor):
        return embs
    if isinstance(embs, tuple):
        return embs[0] if len(embs) > 0 and isinstance(embs[0], torch.Tensor) else embs
    if hasattr(embs, "pooler_output") and embs.pooler_output is not None:
        return embs.pooler_output
    if hasattr(embs, "last_hidden_state") and isinstance(embs.last_hidden_state, torch.Tensor):
        return embs.last_hidden_state[:, 0]
    return embs


def main():
    dataset_modules = discover_dataset_modules()
    parser = argparse.ArgumentParser(description="Generate query/doc embeddings")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=sorted(dataset_modules.keys())
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--encoder_mode", type=str, default="auto")
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        help="Override model pooling (e.g. 'none' for ColBERT token embeddings)",
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
    parser.add_argument("--subset", type=str, default="test")

    base_args, remaining = parser.parse_known_args()
    dataset_module = dataset_modules[base_args.dataset]
    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_module.add_generate_arguments(dataset_parser)
    dataset_args = dataset_parser.parse_args(remaining)

    args = argparse.Namespace(**vars(base_args), **vars(dataset_args))

    accelerator = Accelerator()
    is_main = accelerator.is_local_main_process

    if is_main:
        print(f"Loading model: {args.model_path}")

    processor = None
    image_processor = None

    try:
        # Use encoder_mode for resolution if possible to hit the specific loader
        processor = ProcessorRegistry.resolve(args.model_path, encoder_mode=args.encoder_mode)
    except Exception:
        # Fallback to pure model path resolution
        try:
            processor = ProcessorRegistry.resolve(args.model_path)
        except Exception:
            processor = None
    if processor is None:
        try:
            image_processor = AutoImageProcessor.from_pretrained(args.model_path)
        except (OSError, ValueError):
            image_processor = None
    if processor is not None and image_processor is None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            wrapped = getattr(processor, "processor", None)
            if wrapped is not None:
                image_processor = getattr(wrapped, "image_processor", None)
        if image_processor is None:
            try:
                image_processor = AutoImageProcessor.from_pretrained(args.model_path)
            except (OSError, ValueError):
                image_processor = None

    # Auto-load vembed_config.json from checkpoint (if present)
    vembed_cfg_path = os.path.join(args.model_path, "vembed_config.json")
    vembed_cfg = {}
    if os.path.isfile(vembed_cfg_path):
        import json

        with open(vembed_cfg_path) as f:
            vembed_cfg = json.load(f)
        if is_main:
            print(f"Loaded vembed_config.json: {vembed_cfg}")

    # CLI args override vembed_config.json values
    model_kwargs = {}
    pooling = args.pooling or vembed_cfg.get("pooling_method")
    if pooling:
        model_kwargs["pooling_method"] = pooling
    proj_dim = args.projection_dim or vembed_cfg.get("projection_dim")
    if proj_dim:
        model_kwargs["projection_dim"] = int(proj_dim)
    topk_tokens = args.topk_tokens or vembed_cfg.get("topk_tokens")
    if topk_tokens:
        model_kwargs["topk_tokens"] = int(topk_tokens)

    # Pass encoder_mode to model init
    if args.encoder_mode:
        model_kwargs["encoder_mode"] = args.encoder_mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisualRetrievalModel(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()

    dataset_spec = dataset_module.build_data(args)
    data_source = dataset_spec["data_source"]
    image_root = dataset_spec["image_root"]
    retrieval_mode = dataset_spec["retrieval_mode"]
    extras = dataset_spec.get("extras", {})

    if is_main:
        print(f"Dataset: {args.dataset}")
        print(f"Encoder mode: {args.encoder_mode}")

    dataset = VisualRetrievalDataset(
        data_source=data_source,
        processor=processor,
        image_root=image_root,
        mode="eval",
    )
    collator_cls = CollatorRegistry.get(args.encoder_mode) or CollatorRegistry.get("default")
    # For SigLIP, the processor is monkey-patched to enforce max_length=64.
    # The default collator will use this processor.
    collator = collator_cls(processor=processor, image_processor=image_processor, mode="eval")

    if is_main:
        print(f"Resolved collator: {collator.__class__.__name__}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Use single GPU directly without accelerator for debugging consistency
    feature_model = model

    query_embeddings_all, doc_embeddings_all = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, disable=not is_main, desc="Encoding")):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            q_batch, p_batch = _split_query_pos(batch, retrieval_mode)

            q_out, _ = _encode_batch(model, feature_model, q_batch)
            p_out, _ = _encode_batch(model, feature_model, p_batch)

            q_vec = _maybe_first(q_out)
            p_vec = _maybe_first(p_out)

            # L2 Normalize for SigLIP/CLIP retrieval
            if isinstance(q_vec, torch.Tensor):
                q_vec = F.normalize(q_vec, p=2, dim=-1)
            if isinstance(p_vec, torch.Tensor):
                p_vec = F.normalize(p_vec, p=2, dim=-1)

            query_embeddings_all.append(q_vec.detach().cpu().float().numpy())
            doc_embeddings_all.append(p_vec.detach().cpu().float().numpy())

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        query_embeddings_arr = np.concatenate(query_embeddings_all)
        doc_embeddings_arr = np.concatenate(doc_embeddings_all)

        q_path = os.path.join(args.output_dir, f"{args.subset}_query_embeddings.npy")
        p_path = os.path.join(args.output_dir, f"{args.subset}_doc_embeddings.npy")
        np.save(q_path, query_embeddings_arr)
        np.save(p_path, doc_embeddings_arr)

        print(f"Saved query embeddings: {q_path}  {query_embeddings_arr.shape}")
        print(f"Saved doc embeddings:   {p_path}  {doc_embeddings_arr.shape}")

        for key, value in extras.items():
            extra_path = os.path.join(args.output_dir, f"{args.subset}_{key}.npy")
            np.save(extra_path, value)
            print(f"Saved {key}: {extra_path}  {np.asarray(value).shape}")


if __name__ == "__main__":
    main()
