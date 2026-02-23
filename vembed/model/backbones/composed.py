"""
Composed (dual-encoder) embedding model — **EXPERIMENTAL**

Combines independently pre-trained text and image encoders (e.g. BERT + DINOv2)
with learnable projection heads to align them in a shared embedding space.

.. warning::
    This mode is experimental.  Because the two encoders are pre-trained
    separately without any cross-modal alignment, the projection heads must
    learn the alignment from scratch.  For production use-cases, prefer a
    pre-aligned model such as CLIP / SigLIP or a VLM like Qwen3-VL.
"""

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel, pool
from ..encoders_factory import SimpleImageEncoder, SimpleTextEncoder
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

# Default shared embedding dimension when projection_dim is not set
_DEFAULT_PROJECTION_DIM = 512


@ModelRegistry.register("composed")
class ComposedEmbeddingModel(BaseEmbeddingModel):
    """
    **[Experimental]** Dual-encoder model that combines a text encoder and an
    image encoder through learnable projection heads.

    The projection heads are **always** created (even when both encoders share
    the same hidden size) because the two pre-trained representation spaces are
    not aligned and need a learned mapping.

    Config keys:
        text_model_name:   HuggingFace model id for the text encoder.
        image_model_name:  HuggingFace model id for the image encoder.
        projection_dim:    Shared embedding dimension (default: 512).
        projection_layers: Number of layers in the projector (1 = linear, 2 = MLP). Default: 2.
        pooling_method:    Pooling strategy for text encoder (default: "mean").
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.pooling_method = config.get("pooling_method", "mean")
        self.use_mrl = config.get("use_mrl", False)

        logger.warning(
            "[Experimental] ComposedEmbeddingModel is experimental. "
            "Cross-modal alignment is learned from scratch via projection heads."
        )

        self.text_encoder = SimpleTextEncoder(
            config["text_model_name"], pooling=self.pooling_method, model_config=config
        )
        self.image_encoder = SimpleImageEncoder(
            config["image_model_name"], pooling="cls", model_config=config
        )

        text_hidden = getattr(self.text_encoder.model.config, "hidden_size", 768)
        img_hidden = getattr(self.image_encoder.model.config, "hidden_size", 768)

        self.projection_dim = config.get("projection_dim") or _DEFAULT_PROJECTION_DIM
        proj_layers = int(config.get("projection_layers") or 2)

        self.mrl_dims = config.get("mrl_dims") or [self.projection_dim]
        if self.use_mrl:
            self.mrl_dims = sorted(self.mrl_dims, reverse=True)

        # Always create projection heads — even when dimensions match the two
        # representation spaces are NOT aligned and need a learned projection.
        self.text_projection = self._build_projector(text_hidden, self.projection_dim, proj_layers)
        self.image_projection = self._build_projector(img_hidden, self.projection_dim, proj_layers)

        logger.info(
            f"Composed projectors: text {text_hidden}→{self.projection_dim}, "
            f"image {img_hidden}→{self.projection_dim}, layers={proj_layers}"
        )

        # Try to load pre-trained projection weights if resuming
        self._try_load_projections(config)

    @staticmethod
    def _build_projector(in_dim: int, out_dim: int, num_layers: int = 2) -> nn.Module:
        """Build a projection head (linear or MLP)."""
        in_dim, out_dim, num_layers = int(in_dim), int(out_dim), int(num_layers)
        if num_layers <= 1:
            return nn.Linear(in_dim, out_dim, bias=False)
        # 2-layer MLP: Linear → GELU → Linear
        return nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def _try_load_projections(self, config: dict[str, Any]):
        """Attempt to load saved projection weights from a checkpoint directory."""
        model_path = config.get("model_name_or_path", "")
        if not model_path or not os.path.isdir(model_path):
            return

        text_proj_path = os.path.join(model_path, "text_projection.pt")
        image_proj_path = os.path.join(model_path, "image_projection.pt")

        if os.path.isfile(text_proj_path):
            self.text_projection.load_state_dict(torch.load(text_proj_path, map_location="cpu"))
            logger.info(f"Loaded text projection from {text_proj_path}")

        if os.path.isfile(image_proj_path):
            self.image_projection.load_state_dict(torch.load(image_proj_path, map_location="cpu"))
            logger.info(f"Loaded image projection from {image_proj_path}")

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.text_encoder.model.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.image_encoder.model.save_pretrained(os.path.join(save_directory, "image_encoder"))
        torch.save(
            self.text_projection.state_dict(),
            os.path.join(save_directory, "text_projection.pt"),
        )
        torch.save(
            self.image_projection.state_dict(),
            os.path.join(save_directory, "image_projection.pt"),
        )

    def _project_text(self, embs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projection(embs), dim=-1)

    def _project_image(self, embs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projection(embs), dim=-1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs,
    ):
        # Text only
        if input_ids is not None and pixel_values is None:
            outputs = self.text_encoder.model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            return self._project_text(pool(outputs, attention_mask, self.pooling_method))

        # Image only
        if pixel_values is not None and input_ids is None:
            outputs = self.image_encoder.model(pixel_values=pixel_values, **kwargs)
            return self._project_image(pool(outputs, attention_mask=None, method="cls"))

        # Both (multi-modal)
        if input_ids is not None and pixel_values is not None:
            t_out = self.text_encoder.model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            i_out = self.image_encoder.model(pixel_values=pixel_values, **kwargs)
            return (
                self._project_text(pool(t_out, attention_mask, self.pooling_method)),
                self._project_image(pool(i_out, attention_mask=None, method="cls")),
            )
