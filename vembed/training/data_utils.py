"""Data processing utilities for training."""

from typing import Any

import torch


def maybe_first(embs: Any) -> torch.Tensor:
    """Extract a plain tensor from model output.

    Handles various output formats: torch.Tensor, ModelOutput, tuples.

    Args:
        embs: Model output (can be Tensor, tuple, or ModelOutput with pooler_output).

    Returns:
        Extracted tensor from the output.
    """
    if isinstance(embs, torch.Tensor):
        return embs
    if isinstance(embs, tuple):
        return embs[0] if len(embs) > 0 and isinstance(embs[0], torch.Tensor) else embs
    if hasattr(embs, "pooler_output") and embs.pooler_output is not None:
        return embs.pooler_output
    if hasattr(embs, "last_hidden_state") and isinstance(embs.last_hidden_state, torch.Tensor):
        return embs.last_hidden_state[:, 0]
    return embs


def unpack_query_batch(batch: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    """Extract query inputs from batch based on retrieval mode.

    Args:
        batch: Input batch dictionary.
        retrieval_mode: Retrieval mode (e.g., 't2i', 'i2i', 'm2i', etc.).
            First char: 't'=text, 'i'=image, 'm'=multimodal
            Last char: 't'=text, 'i'=image

    Returns:
        Dictionary with extracted query inputs.

    Raises:
        KeyError: If required keys for the retrieval mode are missing.
    """
    if retrieval_mode.startswith("i"):
        if "query_pixel_values" not in batch:
            raise KeyError(
                f"retrieval_mode={retrieval_mode} requires 'query_pixel_values'. "
                f"Got keys: {list(batch.keys())}"
            )
        result: dict[str, Any] = {"pixel_values": batch["query_pixel_values"]}
        # VLMs (e.g., Qwen3-VL) also need input_ids for image items
        if "query_input_ids" in batch and batch["query_input_ids"] is not None:
            result["input_ids"] = batch["query_input_ids"]
            result["attention_mask"] = batch["query_attention_mask"]
        elif "input_ids" in batch and batch["input_ids"] is not None:
            result["input_ids"] = batch["input_ids"]
            result["attention_mask"] = batch["attention_mask"]
        if "query_image_grid_thw" in batch and batch["query_image_grid_thw"] is not None:
            result["image_grid_thw"] = batch["query_image_grid_thw"]
        return result

    if retrieval_mode.startswith("m"):
        result = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "query_pixel_values" in batch and batch["query_pixel_values"] is not None:
            result["pixel_values"] = batch["query_pixel_values"]
        if "query_image_grid_thw" in batch and batch["query_image_grid_thw"] is not None:
            result["image_grid_thw"] = batch["query_image_grid_thw"]
        return result

    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


def unpack_positive_batch(batch: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    """Extract positive sample inputs from batch based on retrieval mode.

    Args:
        batch: Input batch dictionary.
        retrieval_mode: Retrieval mode.

    Returns:
        Dictionary with extracted positive inputs.
    """
    if retrieval_mode.endswith("t"):
        return {"input_ids": batch["pos_input_ids"], "attention_mask": batch["pos_attention_mask"]}

    # Image positive: prefer prefixed keys, fallback to legacy aliases
    pv = batch.get("pos_pixel_values")
    if pv is None:
        pv = batch.get("pixel_values")
    result: dict[str, Any] = {"pixel_values": pv}

    # VLMs (e.g., Qwen3-VL) also need input_ids for image items
    if "pos_input_ids" in batch and batch["pos_input_ids"] is not None:
        result["input_ids"] = batch["pos_input_ids"]
        result["attention_mask"] = batch["pos_attention_mask"]

    # image_grid_thw for VLMs
    grid = batch.get("pos_image_grid_thw")
    if grid is None:
        grid = batch.get("image_grid_thw")
    if grid is not None:
        result["image_grid_thw"] = grid

    return result


def unpack_negative_batch(batch: dict[str, Any]) -> dict[str, Any] | None:
    """Extract negative sample inputs from batch.

    Args:
        batch: Input batch dictionary.

    Returns:
        Dictionary with extracted negative inputs, or None if no negatives.
    """
    if batch.get("neg_pixel_values") is None:
        return None

    n_inputs = {"pixel_values": batch["neg_pixel_values"]}
    if "neg_input_ids" in batch:
        n_inputs["input_ids"] = batch["neg_input_ids"]
        n_inputs["attention_mask"] = batch["neg_attention_mask"]
    if batch.get("neg_image_grid_thw") is not None:
        n_inputs["image_grid_thw"] = batch["neg_image_grid_thw"]

    return n_inputs


def concat_batches(
    batches: list[dict[str, Any]], pad_token_id: int = 0
) -> tuple[dict[str, Any], list[int]]:
    """Concatenate multiple input batches into a single batch.

    Used for unified models (e.g., Qwen-VL) that tokenize both text and images.

    Args:
        batches: List of batch dictionaries to concatenate.
        pad_token_id: Token ID to use for padding sequences.

    Returns:
        Tuple of:
        - concatenated_batch: Dictionary with concatenated tensors.
        - batch_sizes: List of original batch sizes (to split outputs later).

    Raises:
        ValueError: If batch size cannot be determined from inputs.
    """
    batch_sizes = []
    for b in batches:
        if "input_ids" in b and b["input_ids"] is not None:
            batch_sizes.append(b["input_ids"].size(0))
        elif "pixel_values" in b and b["pixel_values"] is not None:
            batch_sizes.append(b["pixel_values"].size(0))
        else:
            raise ValueError(
                f"Cannot determine batch size: batch keys={list(b.keys())}, "
                f"input_ids is {b.get('input_ids')}, pixel_values is {b.get('pixel_values')}"
            )

    all_keys = set()
    for b in batches:
        all_keys.update(b.keys())

    concatenated = {}

    # Process input_ids and attention_mask (requires padding to max length)
    if "input_ids" in all_keys:
        max_len = 0
        for b in batches:
            if "input_ids" in b and b["input_ids"] is not None:
                max_len = max(max_len, b["input_ids"].size(1))

        padded_ids = []
        padded_masks = []

        for b in batches:
            if "input_ids" in b and b["input_ids"] is not None:
                curr_ids = b["input_ids"]
                curr_mask = b["attention_mask"]
                B, L = curr_ids.shape
                diff = max_len - L
                if diff > 0:
                    pad_tensor = torch.full(
                        (B, diff), pad_token_id, dtype=curr_ids.dtype, device=curr_ids.device
                    )
                    curr_ids = torch.cat([curr_ids, pad_tensor], dim=1)

                    mask_pad = torch.zeros(
                        (B, diff), dtype=curr_mask.dtype, device=curr_mask.device
                    )
                    curr_mask = torch.cat([curr_mask, mask_pad], dim=1)

                padded_ids.append(curr_ids)
                padded_masks.append(curr_mask)

        if padded_ids:
            concatenated["input_ids"] = torch.cat(padded_ids, dim=0)
            concatenated["attention_mask"] = torch.cat(padded_masks, dim=0)

    # Process pixel_values (simple concatenation, no padding needed)
    if "pixel_values" in all_keys:
        pvs = []
        for b in batches:
            pv = b.get("pixel_values")
            if pv is not None:
                pvs.append(pv)
        if pvs:
            concatenated["pixel_values"] = torch.cat(pvs, dim=0)

    # Process image_grid_thw for VLM models (simple concatenation)
    if "image_grid_thw" in all_keys:
        grids = []
        for b in batches:
            g = b.get("image_grid_thw")
            if g is not None:
                grids.append(g)
        if grids:
            concatenated["image_grid_thw"] = torch.cat(grids, dim=0)

    return concatenated, batch_sizes
