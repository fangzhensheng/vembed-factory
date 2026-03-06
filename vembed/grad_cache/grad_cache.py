import logging
from collections import UserDict
from collections.abc import Callable
from contextlib import nullcontext
from itertools import repeat
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """Save and restore RNG state for deterministic recomputation.

    Uses fork_rng for proper scoping across all GPU devices.
    Handles None tensors with fallback to all available devices.
    """

    def __init__(self, *tensors):
        """Save current RNG state for later restoration."""
        self.fwd_cpu_state = torch.get_rng_state()

        try:
            self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)
        except Exception:
            if torch.cuda.is_available():
                self.fwd_gpu_devices = list(range(torch.cuda.device_count()))
                self.fwd_gpu_states = [
                    torch.cuda.get_rng_state(device=i) for i in self.fwd_gpu_devices
                ]
            else:
                self.fwd_gpu_devices = []
                self.fwd_gpu_states = []

        self._fork = None

    def __enter__(self):
        """Restore saved RNG state using fork_rng for proper scoping."""
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()

        torch.set_rng_state(self.fwd_cpu_state)
        if self.fwd_gpu_devices and self.fwd_gpu_states:
            set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous RNG state when exiting scope."""
        if self._fork is not None:
            self._fork.__exit__(exc_type, exc_val, exc_tb)
            self._fork = None


logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache mechanism for large batch training with limited GPU memory.

    Splits inputs into chunks, performs a forward pass without gradients to compute
    representations, computes the loss and gradients w.r.t representations, and then
    performs a second forward-backward pass to propagate gradients to the model.
    """

    def __init__(
        self,
        models: list[nn.Module],
        chunk_sizes: int | list[int],
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] | None = None,
        get_rep_fn: Callable[..., Tensor] | None = None,
        fp16: bool = False,
        scaler: GradScaler | None = None,
    ):
        """
        Initialize the Gradient Cache.

        Args:
            models: List of encoder models to update.
            chunk_sizes: Batch size for each chunk.
            loss_fn: Loss function accepting representation tensors.
                    When using BaseLoss subclasses, gather is handled automatically.
            split_input_fn: Optional custom function to split inputs.
            get_rep_fn: Optional custom function to extract representations from model output.
            fp16: Enable mixed precision training.
            scaler: GradScaler for mixed precision (required if fp16 is True).

        Note:
            When gradient cache is enabled with distributed training, embeddings are
            gathered across processes by BaseLoss subclasses before computing the loss.
            This allows proper in-batch negative sampling across multiple GPUs.
        """
        self.models = models

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        if fp16:
            assert scaler is not None, "Mixed precision training requires a gradient scaler."

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False
        self._inactive_params_cache = {}

    def __call__(self, *args, **kwargs) -> Tensor:
        """Run the cache step."""
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input: Any, chunk_size: int) -> list[Any]:
        """
        Split model input into chunks of size `chunk_size`.

        Handles Tensors, dictionaries of Tensors, and lists of Tensors.
        Delegates to `split_input_fn` if provided.
        """
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        if isinstance(model_input, (dict, UserDict)):
            return self._split_dict_inputs(model_input, chunk_size)

        if isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_tensors = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_tensors)]

        if (
            isinstance(model_input, tuple)
            and len(model_input) == 2
            and isinstance(model_input[0], list)
            and isinstance(model_input[1], dict)
        ):
            # Handle (args, kwargs) tuple
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        raise NotImplementedError(f"Model input split not implemented for type {type(model_input)}")

    def _split_dict_inputs(self, model_input: dict | UserDict, chunk_size: int) -> list[dict]:
        """Helper to split dictionary inputs."""
        tensor_keys = [k for k, v in model_input.items() if isinstance(v, Tensor)]
        non_tensor_keys = [
            k for k, v in model_input.items() if v is not None and not isinstance(v, Tensor)
        ]

        if not tensor_keys:
            if not model_input:
                return []
            raise ValueError("Cannot split input dict with no Tensors.")

        # Determine batch size from the first tensor
        first_tensor = model_input[tensor_keys[0]]
        batch_size = first_tensor.shape[0]
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        chunked_data = [{} for _ in range(num_chunks)]

        # Split Tensors
        for k in tensor_keys:
            chunks = model_input[k].split(chunk_size, dim=0)
            for i, chunk in enumerate(chunks):
                chunked_data[i][k] = chunk

        # Replicate Non-Tensors
        for k in non_tensor_keys:
            val = model_input[k]
            for i in range(num_chunks):
                chunked_data[i][k] = val

        return chunked_data

    def get_input_tensors(self, model_input: Any) -> list[Tensor]:
        """
        Recursively extract all tensors from model input to track random states.
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        if isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        if isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        if self._get_input_tensors_strict:
            raise NotImplementedError(
                f"get_input_tensors not implemented for type {type(model_input)}"
            )

        return []

    def model_call(self, model: nn.Module, model_input: Any) -> Any:
        """Call the model with the given input."""
        with autocast() if self.fp16 else nullcontext():
            if isinstance(model_input, Tensor):
                return model(model_input)

            if isinstance(model_input, list):
                return model(*model_input)

            if isinstance(model_input, (dict, UserDict)):
                return model(**model_input)

            if isinstance(model_input, tuple) and len(model_input) == 2:
                model_args, model_kwargs = model_input
                if isinstance(model_args, list) and isinstance(model_kwargs, dict):
                    return model(*model_args, **model_kwargs)

            raise NotImplementedError(
                f"Model call not implemented for input type {type(model_input)}"
            )

    def get_reps(self, model_out: Any) -> Tensor:
        """Extract representation tensor from model output."""
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)

        # Heuristic: if output is a tuple/list and starts with a Tensor, return the first element.
        # This handles cases where models return (loss, logits) or similar tuples.
        if (
            isinstance(model_out, (tuple, list))
            and len(model_out) > 0
            and isinstance(model_out[0], Tensor)
        ):
            return model_out[0]

        return cast(Tensor, model_out)

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """Compute the loss using the provided loss function."""
        return self.loss_fn(*reps, **loss_kwargs)

    def forward_no_grad(
        self,
        model: nn.Module,
        model_inputs: list[Any],
    ) -> tuple[Tensor, list[RandContext]]:
        """
        Perform the first forward pass without gradients.

        Returns:
            Tuple of (concatenated representations, list of random states).
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for input_chunk in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(input_chunk)))
                output_chunk = self.model_call(model, input_chunk)
                model_reps.append(self.get_reps(output_chunk))

        # Concatenate all sub-batch representations
        concatenated_reps = torch.cat(model_reps, dim=0)
        return concatenated_reps, rnd_states

    def build_cache(self, *reps: Tensor, **loss_kwargs) -> tuple[list[Tensor], Tensor]:
        """
        Compute gradients w.r.t representations (the "cache").

        Returns:
            Tuple of (list of gradient tensors, loss tensor).
        """
        reps_with_grad = [r.detach().requires_grad_() for r in reps]

        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps_with_grad, **loss_kwargs)

        if self.fp16 and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = []
        for r in reps_with_grad:
            if r.grad is None:
                cache.append(torch.zeros_like(r))
            else:
                cache.append(r.grad)

        return cache, loss.detach()

    def _get_device_from_input(self, model_input: Any) -> torch.device:
        """Extract device from model input."""
        if isinstance(model_input, Tensor):
            return model_input.device
        if isinstance(model_input, (dict, UserDict)):
            for v in model_input.values():
                if isinstance(v, Tensor):
                    return v.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_active_encoder_type(self, model_input: Any) -> str | None:
        """Detect which encoder is active based on input structure."""
        if isinstance(model_input, (dict, UserDict)):
            keys_lower = {k.lower() for k in model_input.keys()}

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"GradientCache: Detecting encoder type from input keys: {list(model_input.keys())}"
                )

            text_keywords = {"input_ids", "attention_mask", "token_type_ids", "text"}
            image_keywords = {"pixel_values", "image", "pixel_mask", "patch_mask"}

            if text_keywords & keys_lower:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GradientCache: Detected text encoder")
                return "text"
            if image_keywords & keys_lower:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GradientCache: Detected image encoder")
                return "image"

        if isinstance(model_input, list):
            if model_input and isinstance(model_input[0], Tensor):
                tensor = model_input[0]
                if tensor.dim() == 2 and tensor.dtype in (torch.int64, torch.int32):
                    return "text"
                if tensor.dim() == 4:
                    return "image"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("GradientCache: Could not detect encoder type")
        return None

    def _create_dummy_loss_for_chunk(self, model: nn.Module, model_input: Any) -> Tensor | None:
        """Create fresh dummy loss for a single chunk (avoid graph reuse).

        Must create fresh dummy loss for each chunk to avoid "backward through graph
        twice" error. Using 0.0 instead of tensor(0.0, requires_grad=True) for efficiency.
        """
        active_type = self._get_active_encoder_type(model_input)
        if active_type is None:
            return None

        # Check cache for inactive parameters
        cache_key = (id(model), active_type)
        if cache_key in self._inactive_params_cache:
            inactive_params = self._inactive_params_cache[cache_key]
            dummy_loss = 0.0
            for param in inactive_params:
                dummy_loss = dummy_loss + 0.0 * param.sum()
            return dummy_loss

        if isinstance(model, nn.parallel.DistributedDataParallel):
            underlying_model = model.module
        else:
            underlying_model = model

        # Handle AutoEmbeddingModel wrapper: access backbone directly
        search_root = (
            underlying_model.backbone if hasattr(underlying_model, "backbone") else underlying_model
        )

        # Collect parameters that should be "active" based on input type
        active_params = set()
        if active_type == "text":
            # Text input: mark text-related parameters as active
            if hasattr(search_root, "text_model"):
                for p in search_root.text_model.parameters():
                    active_params.add(id(p))
            if hasattr(search_root, "text_projection"):
                for p in search_root.text_projection.parameters():
                    active_params.add(id(p))
        elif active_type == "image":
            # Image input: mark vision-related parameters as active
            if hasattr(search_root, "vision_model"):
                for p in search_root.vision_model.parameters():
                    active_params.add(id(p))
            if hasattr(search_root, "visual_projection"):
                for p in search_root.visual_projection.parameters():
                    active_params.add(id(p))

        # Create dummy loss for all inactive parameters
        dummy_loss = 0.0
        param_count = 0
        inactive_params = []
        for param in search_root.parameters():
            if param.requires_grad and id(param) not in active_params:
                dummy_loss = dummy_loss + 0.0 * param.sum()
                param_count += 1
                inactive_params.append(param)

        self._inactive_params_cache[cache_key] = inactive_params

        if param_count == 0:
            logger.warning(
                f"GradientCache: No inactive parameters found. active_type={active_type}"
            )
            return None

        return dummy_loss

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs: list[Any],
        cached_gradients: list[Tensor],
        random_states: list[RandContext],
        no_sync_except_last: bool = False,
    ) -> None:
        """Recompute forward and backward with cached gradients."""
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext] * len(model_inputs)

        for input_chunk, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts
        ):
            with state:
                output_chunk = self.model_call(model, input_chunk)
            reps = self.get_reps(output_chunk)

            with sync_context():
                surrogate = torch.dot(reps.flatten(), gradient.flatten())

                dummy_loss = self._create_dummy_loss_for_chunk(model, input_chunk)
                if dummy_loss is not None:
                    surrogate = surrogate + dummy_loss

                surrogate.backward()

    def cache_step(self, *model_inputs, no_sync_except_last: bool = False, **loss_kwargs) -> Tensor:
        """Run a full Gradient Cache step."""
        if no_sync_except_last:
            if not all(isinstance(m, nn.parallel.DistributedDataParallel) for m in self.models):
                raise ValueError("no_sync_except_last requires all models to be wrapped in DDP.")

        chunked_inputs = [
            self.split_inputs(inp, chunk_size)
            for inp, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        all_reps = []
        all_rnd_states = []

        for model, inputs in zip(self.models, chunked_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, inputs)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        chunked_cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, inputs, model_cache, rnd_states in zip(
            self.models, chunked_inputs, chunked_cache, all_rnd_states
        ):
            self.forward_backward(
                model,
                inputs,
                list(model_cache),
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss
