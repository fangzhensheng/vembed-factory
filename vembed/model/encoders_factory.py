from typing import Any

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from .base import pool, resolve_pretrained_kwargs


class SimpleTextEncoder(nn.Module):
    """Text encoder wrapper around HuggingFace AutoModel."""

    def __init__(
        self, model_name: str, pooling: str = "mean", model_config: dict[str, Any] | None = None
    ):
        super().__init__()
        load_kwargs = resolve_pretrained_kwargs(model_config or {})
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        self.pooling = pooling

    def encode(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Encode a list of texts into embeddings."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            self.model.to(device)
        outputs = self.model(**inputs)
        return pool(outputs, inputs.get("attention_mask"), self.pooling)


class SimpleImageEncoder(nn.Module):
    """Image encoder wrapper around HuggingFace AutoModel."""

    def __init__(
        self, model_name: str, pooling: str = "cls", model_config: dict[str, Any] | None = None
    ):
        super().__init__()
        load_kwargs = resolve_pretrained_kwargs(model_config or {})

        # Note: AutoModel handles both ResNet and MAE architectures correctly
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.pooling = pooling

    def encode(self, pil_images: list, device: torch.device | None = None) -> torch.Tensor:
        """Encode a list of PIL images into embeddings."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if device:
            pixel_values = pixel_values.to(device)
            self.model.to(device)

        outputs = self.model(pixel_values=pixel_values)

        # pool() function handles different output types (last_hidden_state, pooler_output)
        # automatically, so no explicit architecture checks are needed here.
        return pool(outputs, attention_mask=None, method=self.pooling)
