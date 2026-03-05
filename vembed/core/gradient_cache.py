"""Gradient Cache integration with vembed-factory pipeline."""

from collections import UserDict

from torch import Tensor

from vembed.grad_cache import GradCache as LibGradCache


def _extract_rep(output: object) -> Tensor:
    """Extract plain tensor from model output."""
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
    """Custom split for VLM inputs (e.g. Qwen3-VL).

    Handles pixel_values (flat patches) and image_grid_thw (per-image metadata).
    Splits pixel_values based on patch counts derived from grid_thw.
    """
    if isinstance(model_input, Tensor):
        return list(model_input.split(chunk_size, dim=0))

    if not isinstance(model_input, dict | UserDict):
        raise NotImplementedError(f"_split_vlm_inputs not implemented for type {type(model_input)}")

    has_pixel_values = "pixel_values" in model_input and model_input["pixel_values"] is not None
    has_grid = "image_grid_thw" in model_input and model_input["image_grid_thw"] is not None

    if not (has_pixel_values and has_grid):
        # Standard split
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

    # Batch size from input_ids or attention_mask (preferred)
    if "input_ids" in model_input:
        batch_size = model_input["input_ids"].shape[0]
    elif "attention_mask" in model_input:
        batch_size = model_input["attention_mask"].shape[0]
    else:
        batch_size = grid_thw.shape[0]

    patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

    n_chunks = (batch_size + chunk_size - 1) // chunk_size
    result = []
    px_offset = 0

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, batch_size)

        chunk_dict = {}

        for k in ("input_ids", "attention_mask"):
            if k in model_input and model_input[k] is not None:
                chunk_dict[k] = model_input[k][start:end]

        chunk_dict["image_grid_thw"] = grid_thw[start:end]

        chunk_n_patches = sum(patches_per_image[start:end])
        chunk_dict["pixel_values"] = pixel_values[px_offset : px_offset + chunk_n_patches]
        px_offset += chunk_n_patches

        result.append(chunk_dict)

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

        if mode.endswith("t"):
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                p["attention_mask"] = batch["pos_attention_mask"]
        else:
            # Prefer prefixed keys
            pv = batch.get("pos_pixel_values") or batch.get("pixel_values")
            if pv is not None:
                p["pixel_values"] = pv
            
            grid = batch.get("pos_image_grid_thw") or batch.get("image_grid_thw")
            if grid is not None:
                p["image_grid_thw"] = grid
            
            # VLM image items need input_ids (placeholder tokens)
            if "pos_input_ids" in batch:
                p["input_ids"] = batch["pos_input_ids"]
                p["attention_mask"] = batch["pos_attention_mask"]

        if "neg_pixel_values" in batch:
            n["pixel_values"] = batch["neg_pixel_values"]
            if "neg_image_grid_thw" in batch:
                n["image_grid_thw"] = batch["neg_image_grid_thw"]
            if "neg_input_ids" in batch:
                n["input_ids"] = batch["neg_input_ids"]
                n["attention_mask"] = batch["neg_attention_mask"]

        return q, p, n

    def step(self, model, batch) -> float:
        q_batch, p_batch, n_batch = self._unpack_batch(batch)

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
