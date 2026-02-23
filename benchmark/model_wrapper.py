"""
Adapts VisualRetrievalModel to a uniform encode_text / encode_image interface
for benchmark evaluation.
"""

import logging
import os
import sys
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from PIL import Image

# Resolve project root so vembed is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry

logger = logging.getLogger(__name__)


class ModelWrapper(ABC):
    """Uniform interface for any model that encodes text and images."""

    @abstractmethod
    def encode_text(self, texts: list[str], device: str) -> torch.Tensor: ...

    @abstractmethod
    def encode_image(self, images: list[Image.Image], device: str) -> torch.Tensor: ...


class VEmbedWrapper(ModelWrapper):
    """Wraps a trained VisualRetrievalModel checkpoint for benchmarking."""

    def __init__(self, model_path: str):
        self.processor = self._load_processor(model_path)
        self.model = VisualRetrievalModel(model_path)
        self.model.eval()

    @staticmethod
    def _load_processor(model_path: str):
        return ProcessorRegistry.resolve(model_path)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: str) -> torch.Tensor:
        self.model.to(device)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, p=2, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(self, images: list[Image.Image], device: str) -> torch.Tensor:
        self.model.to(device)
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats, p=2, dim=-1).cpu()
