"""Qwen3-Embedding processor loader.

Uses ``AutoTokenizer`` instead of ``AutoProcessor`` since Qwen3-Embedding
is a text-only model.
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from .registry import BaseProcessorLoader, ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("qwen3_embedding")
class Qwen3EmbeddingProcessorLoader(BaseProcessorLoader):
    """Loads the tokenizer for Qwen3-Embedding models."""

    @staticmethod
    def match(model_name: str) -> bool:
        lower = model_name.lower()
        return "qwen3-embedding" in lower or "qwen3_embedding" in lower

    @staticmethod
    def load(model_name: str, **kwargs):
        kwargs.setdefault("padding_side", "left")
        kwargs.setdefault("trust_remote_code", True)
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
