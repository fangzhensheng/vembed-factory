"""Gradient Cache integration with vembed-factory pipeline."""

import logging
from collections import UserDict

from torch import Tensor

from vembed.core.constants import (
    BATCH_SIZE_PRIORITY_KEYS,
    GRID_INDICATOR,
    GRID_TO_PATCH_MAP,
    PATCH_INDICATORS,
)
from vembed.grad_cache import GradCache as LibGradCache

logger = logging.getLogger(__name__)


def _extract_rep(output: object) -> Tensor:
    """Extract plain tensor from model output."""
    # DEBUG: Log output type before checking
    if output is None:
        logger.warning(f"[EXTRACT_REP] model returned None!")
    else:
        logger.warning(f"[EXTRACT_REP] output type: {type(output)}, has pooler_output: {hasattr(output, 'pooler_output')}, has last_hidden_state: {hasattr(output, 'last_hidden_state')}")

    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
        return output[0]
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, Tensor):
        return output.last_hidden_state[:, 0]

    logger.error(f"Model output type: {type(output)}")
    if output is None:
        logger.error("Model returned None - check if model forward() is returning values correctly")

    raise TypeError(
        f"Cannot extract tensor from {type(output).__name__}; "
        "expected Tensor, tuple[Tensor], or ModelOutput with pooler_output/last_hidden_state. "
        "Model returned None - check if model is initialized and forward pass is correct."
    )


def _find_batch_size_and_keys(model_input: dict) -> tuple[int | None, list[str]]:
    """Determine the batch size and identify batch-aligned keys."""
    batch_size = None
    batch_aligned_keys = []

    for priority_key in BATCH_SIZE_PRIORITY_KEYS:
        if priority_key in model_input:
            v = model_input[priority_key]
            if isinstance(v, Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                batch_aligned_keys.append(priority_key)
                break

    if batch_size is None:
        for k, v in model_input.items():
            if GRID_INDICATOR in k and isinstance(v, Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                batch_aligned_keys.append(k)
                break

    if batch_size is None:
        for k, v in model_input.items():
            if isinstance(v, Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                batch_aligned_keys.append(k)
                break

    if batch_size is not None:
        for k, v in model_input.items():
            if (
                k not in batch_aligned_keys
                and isinstance(v, Tensor)
                and v.ndim > 0
                and v.shape[0] == batch_size
            ):
                batch_aligned_keys.append(k)

    return batch_size, batch_aligned_keys


def _find_grid_and_patch_keys(model_input: dict) -> tuple[str | None, str | None]:
    """Find the keys corresponding to grid metadata and flat patch tensors."""
    for k in model_input:
        if GRID_INDICATOR in k and isinstance(model_input[k], Tensor):
            expected_patch_key = GRID_TO_PATCH_MAP.get(k)
            if expected_patch_key and expected_patch_key in model_input:
                return k, expected_patch_key

            for pk in model_input:
                if any(ind in pk for ind in PATCH_INDICATORS) and isinstance(
                    model_input[pk], Tensor
                ):
                    return k, pk
            return k, None
    return None, None


def _split_vlm_inputs(model_input, chunk_size: int) -> list:
    """Custom split for VLM inputs with automatic field routing.

    Supports both legacy flat-patch models (pixel_values + image_grid_thw)
    and future hierarchical models (video_grid_thw, etc).

    Uses shape heuristics to route fields:
    - Batch-aligned tensors (shape[0] == batch_size): split on dim 0
    - Per-image metadata (contains grid info): split based on grid counts
    - Scalar/non-tensor fields: replicate across chunks
    """
    if isinstance(model_input, Tensor):
        return list(model_input.split(chunk_size, dim=0))

    if not isinstance(model_input, dict | UserDict):
        raise NotImplementedError(f"_split_vlm_inputs not implemented for type {type(model_input)}")

    batch_size, batch_aligned_keys = _find_batch_size_and_keys(model_input)
    if batch_size is None:
        return [model_input] if model_input else []

    grid_key, flat_patch_key = _find_grid_and_patch_keys(model_input)

    n_chunks = (batch_size + chunk_size - 1) // chunk_size
    result = [{} for _ in range(n_chunks)]

    if grid_key and flat_patch_key:
        grid_thw = model_input[grid_key]
        flat_patches = model_input[flat_patch_key]
        patches_per_item = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

        px_offset = 0
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, batch_size)

            for k in batch_aligned_keys:
                result[ci][k] = model_input[k][start:end]

            result[ci][grid_key] = grid_thw[start:end]
            chunk_n_patches = sum(patches_per_item[start:end])
            result[ci][flat_patch_key] = flat_patches[px_offset : px_offset + chunk_n_patches]
            px_offset += chunk_n_patches

            for k, v in model_input.items():
                if k not in batch_aligned_keys and k != grid_key and k != flat_patch_key:
                    result[ci][k] = v

        return result

    for k in batch_aligned_keys:
        for i, chunk in enumerate(model_input[k].split(chunk_size, dim=0)):
            result[i][k] = chunk

    for k, v in model_input.items():
        if k not in batch_aligned_keys:
            for i in range(n_chunks):
                result[i][k] = v

    return result


class GradientCache:
    """Wraps GradCache with batch unpacking logic."""

    def __init__(self, loss_fn, chunk_size: int, accelerator=None, retrieval_mode: str = "t2i"):
        self.loss_fn = loss_fn
        self.chunk_size = chunk_size
        self.accelerator = accelerator
        self.retrieval_mode = retrieval_mode

    def _unpack_batch(self, batch):
        """Split batch into query, positive, negative dicts based on retrieval mode."""
        q, p, n = {}, {}, {}
        mode = self.retrieval_mode

        if mode.startswith("i"):
            if "query_pixel_values" in batch:
                q["pixel_values"] = batch["query_pixel_values"]
                if "query_image_grid_thw" in batch:
                    q["image_grid_thw"] = batch["query_image_grid_thw"]
        elif mode.startswith("m"):
            q = {k: batch[k] for k in ("input_ids", "attention_mask") if k in batch}
            if "query_pixel_values" in batch:
                q["pixel_values"] = batch["query_pixel_values"]
                if "query_image_grid_thw" in batch:
                    q["image_grid_thw"] = batch["query_image_grid_thw"]
        else:
            q = {k: batch[k] for k in ("input_ids", "attention_mask") if k in batch}
            # For CLIP in t2i mode, query should only be text, not image
            # Remove the code that incorrectly injects pixel_values into query

        if mode.endswith("t"):
            logger.warning(f"[UNPACK] mode ends with 't', treating positive as text")
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                if "pos_attention_mask" in batch:
                    p["attention_mask"] = batch["pos_attention_mask"]
        else:
            logger.warning(f"[UNPACK] mode doesn't end with 't', treating positive as image")
            # Prefer prefixed keys
            pv = batch.get("pos_pixel_values")
            logger.warning(f"[UNPACK] pos_pixel_values in batch: {pv is not None}")
            if pv is None:
                pv = batch.get("pixel_values")
                logger.warning(f"[UNPACK] fallback to pixel_values: {pv is not None}")
            if pv is not None:
                p["pixel_values"] = pv
                logger.warning(f"[UNPACK] set p[pixel_values]")

            grid = batch.get("pos_image_grid_thw")
            if grid is None:
                grid = batch.get("image_grid_thw")
            if grid is not None:
                p["image_grid_thw"] = grid

            # VLM image items need input_ids (placeholder tokens)
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                if "pos_attention_mask" in batch:
                    p["attention_mask"] = batch["pos_attention_mask"]

        if "neg_pixel_values" in batch:
            n["pixel_values"] = batch["neg_pixel_values"]
            if "neg_image_grid_thw" in batch:
                n["image_grid_thw"] = batch["neg_image_grid_thw"]
            if "neg_input_ids" in batch:
                n["input_ids"] = batch["neg_input_ids"]
                if "neg_attention_mask" in batch:
                    n["attention_mask"] = batch["neg_attention_mask"]

        return q, p, n

    def step(self, model, batch) -> float:
        q_batch, p_batch, n_batch = self._unpack_batch(batch)

        # DEBUG: Log unpacked batches
        logger.warning(f"[STEP] q_batch keys: {list(q_batch.keys())}, p_batch keys: {list(p_batch.keys())}")

        loss_kwargs = {}
        if "labels" in batch and batch["labels"] is not None:
            loss_kwargs["labels"] = batch["labels"]

        inputs = [q_batch, p_batch]
        models = [model, model]
        if n_batch:
            inputs.append(n_batch)
            models.append(model)

        gc = LibGradCache(
            models=models,
            chunk_sizes=self.chunk_size,
            loss_fn=self.loss_fn,
            split_input_fn=_split_vlm_inputs,
            get_rep_fn=_extract_rep,
            fp16=False,
            scaler=None,
        )

        use_no_sync = False
        if self.accelerator and self.accelerator.num_processes > 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            is_fsdp = isinstance(model, FSDP) or (
                hasattr(model, "module") and isinstance(model.module, FSDP)
            )
            use_no_sync = not is_fsdp

        loss = gc.cache_step(*inputs, no_sync_except_last=use_no_sync, **loss_kwargs)
        return loss.item()
