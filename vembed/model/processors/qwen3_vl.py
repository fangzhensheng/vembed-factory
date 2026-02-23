"""Qwen3-VL processor loader.

Uses ``Qwen3VLProcessor`` directly instead of ``AutoProcessor`` to avoid
the video-processor auto-detection bug in certain transformers versions.
"""

from __future__ import annotations

import logging
from typing import Any

from .registry import BaseProcessorLoader, ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("qwen3_vl")
class Qwen3VLProcessorLoader(BaseProcessorLoader):
    """Loads the processor for Qwen3-VL / Qwen3-VL-Embedding models."""

    @staticmethod
    def match(model_name: str) -> bool:
        lower = model_name.lower()
        return "qwen3-vl" in lower or "qwen3_vl" in lower

    @staticmethod
    def load(model_name: str, **kwargs: Any):
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        kwargs.setdefault("padding_side", "right")
        # Ensure we don't set default prompt here, let collator handle it (or use empty)
        return Qwen3VLProcessor.from_pretrained(model_name, **kwargs)
