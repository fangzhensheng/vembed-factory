import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


def resolve_pretrained_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for AutoModel.from_pretrained() from the unified config.

    Handles attn_implementation (flash_attention_2 / sdpa / eager), dtype (formerly torch_dtype), and caching.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": True}

    # Attention implementation
    attn_impl = config.get("attn_implementation")
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash_attention_2 requested but flash-attn is not installed. "
                "Falling back to sdpa. Install with: pip install flash-attn --no-build-isolation"
            )
            attn_impl = "sdpa"
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    raw_dtype = config.get("torch_dtype")
    if raw_dtype:
        kwargs["dtype"] = _TORCH_DTYPE_MAP.get(str(raw_dtype), raw_dtype)
    elif attn_impl in ("flash_attention_2", "sdpa"):
        # flash_attention_2 requires float16 or bfloat16
        kwargs["dtype"] = torch.bfloat16

    return kwargs


def disable_kv_cache(model) -> None:
    """Disable KV cache to prevent conflicts with gradient checkpointing.

    Sets use_cache=False after loading for models that support it.
    Avoids initialization errors for models rejecting use_cache parameter.
    """
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def _extract_hidden_state(outputs):
    """Pull the last-layer hidden state from various HuggingFace output formats."""
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, tuple):
        return outputs[0]
    raise AttributeError(f"Cannot extract hidden states from {type(outputs).__name__}")


def pool(outputs, attention_mask, method: str = "cls"):
    """Unified pooling over encoder outputs.

    Supports: cls (default), mean, last_token, none.
    Falls back to pooler_output when available (except for ``none``).

    When ``method="none"`` the full token-level hidden states ``[B, L, D]``
    are returned without any pooling — required for late-interaction models
    like ColBERT.
    """
    if method != "none" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    hidden = _extract_hidden_state(outputs)

    if method == "none":
        return hidden  # [B, L, D] — no pooling

    if method == "cls":
        return hidden[:, 0]

    if method == "last_token":
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_idx, seq_lengths]
        return hidden[:, -1]

    # Default: mean pooling (mask-aware)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return hidden.mean(dim=1)


class BaseEmbeddingModel(nn.Module, ABC):
    """Abstract base for all VEmbed embedding backends."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple: ...

    @abstractmethod
    def save_pretrained(self, save_directory: str): ...
