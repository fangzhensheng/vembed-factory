"""
Adapts VisualRetrievalModel to a uniform encode_text / encode_image interface
for benchmark evaluation.

Supports all model types including:
- Text-only models (qwen, bge, bert, e5, etc.)
- Vision-only models (dinov2, dino, etc.)
- Vision-Language models (qwen-vl, llava, clip, siglip, etc.)
- Composed models (DINOv2+BERT, etc.)
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

    @property
    @abstractmethod
    def supports_images(self) -> bool: ...


def _auto_detect_encoder_mode(model_path: str) -> str | None:
    """Auto-detect encoder_mode from model path.

    Examples:
        "experiments/output_qwen3_vl_embedding_2b" -> "qwen-vl"
        "Qwen/Qwen3-VL-Embedding-2B" -> "qwen-vl"
        "experiments/output_qwen3_embedding" -> "qwen"
        "Qwen/Qwen3-Embedding-8B" -> "qwen"
    """
    lower = model_path.lower()

    # Check for Qwen-VL (multimodal)
    if "qwen" in lower and ("vl" in lower or "vision" in lower or "m3" in lower):
        return "qwen-vl"

    # Check for Qwen text embedding (text-only)
    if "qwen" in lower and "embedding" in lower:
        return "qwen"

    # Check for SigLIP
    if "siglip" in lower:
        return "siglip"

    # Let VisualRetrievalModel auto-detect
    return None


class VEmbedWrapper(ModelWrapper):
    """Wraps a trained VisualRetrievalModel checkpoint for benchmarking.

    Supports all encoder_mode values: auto, clip, siglip, dinov2, bge, bert, e5,
    qwen (text), qwen-vl, llava, composed, etc.
    """

    def __init__(
        self,
        model_path: str,
        encoder_mode: str | None = None,
        attn_implementation: str | None = None,
        torch_dtype: str | None = None,
    ):
        # Auto-detect encoder_mode if not provided
        if encoder_mode is None:
            encoder_mode = _auto_detect_encoder_mode(model_path)
            if encoder_mode:
                logger.info(f"Auto-detected encoder_mode: {encoder_mode}")

        # Load processor with encoder_mode hint
        self.processor = self._load_processor(model_path, encoder_mode)

        # Load model with encoder_mode
        self.model = VisualRetrievalModel(
            model_path,
            encoder_mode=encoder_mode or "auto",
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

        # Detect if model supports images (text-only vs multimodal)
        self._supports_images = self._detect_image_support()

    @staticmethod
    def _load_processor(model_path: str, encoder_mode: str | None = None):
        """Load processor with encoder_mode hint."""
        if encoder_mode:
            # Try to get specific processor loader
            from vembed.model.processors.registry import ProcessorRegistry
            loader_cls = ProcessorRegistry.get(encoder_mode)
            if loader_cls:
                return loader_cls.load(model_path, trust_remote_code=True)
        return ProcessorRegistry.resolve(model_path)

    def _detect_image_support(self) -> bool:
        """Detect if model supports image inputs."""
        # Check if processor has image_processor
        if hasattr(self.processor, "image_processor"):
            return True
        # Check for common vision attributes
        if hasattr(self.model, "backend"):
            backend = self.model.backend
            # Check forward signature for pixel_values
            import inspect
            sig = inspect.signature(backend.forward)
            return "pixel_values" in sig.parameters
        # Default to True for multimodal models
        return True

    @property
    def supports_images(self) -> bool:
        return self._supports_images

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: str) -> torch.Tensor:
        """Encode text inputs to embeddings.

        Uses the unified forward() interface which works for all model types.
        """
        self.model.to(device)

        # Prepare inputs through processor
        if hasattr(self.processor, "tokenizer"):
            # HuggingFace-style processor with separate tokenizer
            inputs = self.processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            # Unified processor
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use forward() interface (works for all models)
        feats = self.model(**inputs)

        # Handle tuple outputs
        if isinstance(feats, tuple):
            feats = feats[0]

        return F.normalize(feats, p=2, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(self, images: list[Image.Image], device: str) -> torch.Tensor:
        """Encode image inputs to embeddings.

        Only works for models that support vision inputs.
        """
        if not self._supports_images:
            raise RuntimeError(
                f"Model at {self.model.config_dict.get('model_name_or_path')} "
                "does not support image encoding (text-only model)."
            )

        self.model.to(device)

        # Prepare inputs through processor
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use forward() interface with pixel_values
        # Need to also pass input_ids if model is a VLM
        kwargs = {"pixel_values": inputs.get("pixel_values")}
        if "input_ids" in inputs:
            kwargs["input_ids"] = inputs["input_ids"]
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"]
        if "image_grid_thw" in inputs:
            kwargs["image_grid_thw"] = inputs["image_grid_thw"]

        feats = self.model(**kwargs)

        # Handle tuple outputs
        if isinstance(feats, tuple):
            feats = feats[0]

        return F.normalize(feats, p=2, dim=-1).cpu()


class TextOnlyWrapper(ModelWrapper):
    """Wrapper for text-only models (qwen3_embedding, bge, etc.).

    encode_image() raises NotImplementedError for these models.
    """

    def __init__(self, model_path: str, encoder_mode: str | None = None):
        if encoder_mode is None:
            encoder_mode = _auto_detect_encoder_mode(model_path)
            if encoder_mode and encoder_mode != "qwen3_embedding":
                logger.warning(f"Detected encoder_mode={encoder_mode}, forcing text-only mode")

        self.processor = ProcessorRegistry.resolve(model_path)
        self.model = VisualRetrievalModel(
            model_path,
            encoder_mode=encoder_mode or "auto",
        )
        self.model.eval()

    @property
    def supports_images(self) -> bool:
        return False

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: str) -> torch.Tensor:
        self.model.to(device)

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        feats = self.model(**inputs)
        if isinstance(feats, tuple):
            feats = feats[0]

        return F.normalize(feats, p=2, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(self, images: list[Image.Image], device: str) -> torch.Tensor:
        raise NotImplementedError(
            "This is a text-only model. Use VEmbedWrapper with a multimodal model "
            "for image encoding."
        )
