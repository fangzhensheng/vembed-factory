import logging

import torch

from ..base import pool
from ..registry import ModelRegistry
from .auto import AutoEmbeddingModel

logger = logging.getLogger(__name__)


@ModelRegistry.register("vision_generic")
@ModelRegistry.register("vision_only")
@ModelRegistry.register("dino")
@ModelRegistry.register("dinov2")
@ModelRegistry.register("dinov3")
@ModelRegistry.register("mae")
@ModelRegistry.register("vit")
class VisionGenericEmbeddingModel(AutoEmbeddingModel):
    """
    Dedicated backend for vision-only models (e.g., DINOv2, MAE, ViT).
    Inherits initialization and projection logic from AutoEmbeddingModel,
    but restricts forward passes to image features only.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            raise ValueError(f"{self.__class__.__name__}.forward() requires 'pixel_values'.")

        if input_ids is not None:
            logger.warning(
                f"{self.__class__.__name__} received 'input_ids' but it is a vision-only model. Ignoring text inputs."
            )

        # Forward pass through the vision backbone
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)

        # Pool the outputs
        emb = pool(outputs, attention_mask=None, method=self.pooling_method)

        # Apply top-K token selection if requested (for late-interaction/ColBERT)
        if self.pooling_method == "none":
            emb = self._select_tokens(emb)

        # Apply optional projection head
        return self._project(emb)

    def get_text_features(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support text encoding. "
            "Please check your retrieval_mode or model configuration."
        )

    def get_image_features(self, pixel_values=None, **kwargs):
        return self.forward(pixel_values=pixel_values, **kwargs)
