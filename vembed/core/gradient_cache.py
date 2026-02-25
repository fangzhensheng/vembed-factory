from collections import UserDict

from torch import Tensor

from vembed.grad_cache import GradCache as LibGradCache


def _extract_rep(output: object) -> Tensor:
    """Extract a plain tensor from model output (handles ModelOutput, tuples)."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
        return output[0]
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, Tensor):
        return output.last_hidden_state[:, 0]
    raise TypeError(
        f"Cannot extract tensor from {type(output).__name__}; "
        "expected Tensor, tuple[Tensor], or ModelOutput with pooler_output/last_hidden_state"
    )


def _split_vlm_inputs(model_input, chunk_size: int) -> list:
    """
    Custom split function for VLM inputs (e.g. Qwen3-VL).

    In VLMs, ``pixel_values`` is a flat tensor of ALL image patches across the
    batch: ``[total_patches, C, H, W]``.  Its dim-0 is *not* the batch size,
    so the default ``split(chunk_size, dim=0)`` breaks the correspondence with
    ``image_grid_thw`` and ``input_ids``.

    This function:
    1. Determines the *batch size* from ``input_ids`` / ``attention_mask``.
    2. Splits ``input_ids``, ``attention_mask``, and ``image_grid_thw`` normally
       on dim-0 by ``chunk_size``.
    3. Splits ``pixel_values`` according to the number of patches that belong
       to each image (computed from ``image_grid_thw``).
    """
    if isinstance(model_input, Tensor):
        return list(model_input.split(chunk_size, dim=0))

    if not isinstance(model_input, dict | UserDict):
        raise NotImplementedError(f"_split_vlm_inputs not implemented for type {type(model_input)}")

    has_pixel_values = "pixel_values" in model_input and model_input["pixel_values"] is not None
    has_grid = "image_grid_thw" in model_input and model_input["image_grid_thw"] is not None

    if not (has_pixel_values and has_grid):
        # Standard split – all tensors share dim-0 = batch_size
        tensor_keys = [k for k, v in model_input.items() if isinstance(v, Tensor)]
        if not tensor_keys:
            return [model_input] if model_input else []
        first = model_input[tensor_keys[0]]
        n_chunks = (first.shape[0] + chunk_size - 1) // chunk_size
        result = [{} for _ in range(n_chunks)]
        for k in tensor_keys:
            for i, chunk in enumerate(model_input[k].split(chunk_size, dim=0)):
                result[i][k] = chunk
        return result

    grid_thw = model_input["image_grid_thw"]  # [num_images, 3]
    pixel_values = model_input["pixel_values"]  # [total_patches, ...]

    # Determine batch size from input_ids (preferred) or image_grid_thw
    if "input_ids" in model_input:
        batch_size = model_input["input_ids"].shape[0]
    elif "attention_mask" in model_input:
        batch_size = model_input["attention_mask"].shape[0]
    else:
        batch_size = grid_thw.shape[0]

    # Patches per image: t * h * w
    patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

    n_chunks = (batch_size + chunk_size - 1) // chunk_size
    result = []
    px_offset = 0

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, batch_size)

        chunk_dict = {}

        # Regular tensors that share dim-0 = batch_size
        for k in ("input_ids", "attention_mask"):
            if k in model_input and model_input[k] is not None:
                chunk_dict[k] = model_input[k][start:end]

        # image_grid_thw – one row per image, same indexing as batch
        chunk_dict["image_grid_thw"] = grid_thw[start:end]

        # pixel_values – variable number of patches per image
        chunk_n_patches = sum(patches_per_image[start:end])
        chunk_dict["pixel_values"] = pixel_values[px_offset : px_offset + chunk_n_patches]
        px_offset += chunk_n_patches

        result.append(chunk_dict)

    return result


class GradientCache:
    """Wraps GradCache with vembed's batch unpacking and retrieval mode logic."""

    def __init__(self, loss_fn, chunk_size: int, accelerator=None, retrieval_mode: str = "t2i"):
        self.loss_fn = loss_fn
        self.chunk_size = chunk_size
        self.accelerator = accelerator
        self.retrieval_mode = retrieval_mode

    def _unpack_batch(self, batch):
        """Split a training batch into query / positive / (optional) negative dicts."""
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

        if mode.endswith("t"):
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                p["attention_mask"] = batch["pos_attention_mask"]
        else:
            # Pixel values: prefer prefixed key, fallback to legacy alias
            pv = batch.get("pos_pixel_values")
            if pv is None:
                pv = batch.get("pixel_values")
            if pv is not None:
                p["pixel_values"] = pv
            # image_grid_thw: prefer prefixed key, fallback to legacy alias
            grid = batch.get("pos_image_grid_thw")
            if grid is None:
                grid = batch.get("image_grid_thw")
            if grid is not None:
                p["image_grid_thw"] = grid
            # For VLMs (e.g. Qwen3-VL) image items also need input_ids
            # because images are represented as placeholder tokens in the text sequence
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                p["attention_mask"] = batch["pos_attention_mask"]

        if "neg_pixel_values" in batch:
            n["pixel_values"] = batch["neg_pixel_values"]
            if "neg_image_grid_thw" in batch:
                n["image_grid_thw"] = batch["neg_image_grid_thw"]
            # For VLMs: negatives also need input_ids if available
            if "neg_input_ids" in batch:
                n["input_ids"] = batch["neg_input_ids"]
                n["attention_mask"] = batch["neg_attention_mask"]

        return q, p, n

    def step(self, model, batch) -> float:
        q_batch, p_batch, n_batch = self._unpack_batch(batch)

        # Extract loss kwargs (e.g. labels for false-negative masking)
        loss_kwargs = {}
        if "labels" in batch and batch["labels"] is not None:
            loss_kwargs["labels"] = batch["labels"]

        inputs = [q_batch, p_batch]
        models = [model, model]
        if n_batch:
            inputs.append(n_batch)
            models.append(model)

        fp16 = False
        scaler = None
        if self.accelerator:
            mixed = self.accelerator.mixed_precision
            if mixed == "fp16":
                fp16 = True
                scaler = getattr(self.accelerator, "scaler", None)
            # bf16 autocast is handled by Accelerate — no scaler needed

        gc = LibGradCache(
            models=models,
            chunk_sizes=self.chunk_size,
            loss_fn=self.loss_fn,
            split_input_fn=_split_vlm_inputs,
            get_rep_fn=_extract_rep,
            fp16=fp16,
            scaler=scaler,
        )

        no_sync = bool(self.accelerator and self.accelerator.num_processes > 1)
        loss = gc.cache_step(*inputs, no_sync_except_last=no_sync, **loss_kwargs)
        return loss.item()
