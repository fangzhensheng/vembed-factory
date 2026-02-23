import json
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from .model.modeling import VisualRetrievalModel
from .model.processors import ProcessorRegistry, build_image_processor, build_text_processor

logger = logging.getLogger(__name__)


class VEmbedFactoryPredictor:
    """Inference engine for vembed-factory models.

    Loads a trained checkpoint and encodes text/images into embeddings.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        encoder_mode: str = "auto",
        text_model_name: str | None = None,
        image_model_name: str | None = None,
        pooling_method: str = "mean",
        mrl_dim: int | None = None,
    ):
        self.device = device
        self.mrl_dim = mrl_dim
        self.encoder_mode = encoder_mode

        logger.info("Loading model from %s", model_path)

        # Load vembed specific config if exists to ensure consistent inference
        config_path = os.path.join(model_path, "vembed_config.json")
        model_kwargs: dict[str, Any] = {
            "pooling_method": pooling_method,
        }

        if os.path.exists(config_path):
            with open(config_path) as f:
                vembed_config = json.load(f)

            # If user didn't override default pooling ("mean"), use config
            if pooling_method == "mean" and "pooling_method" in vembed_config:
                model_kwargs["pooling_method"] = vembed_config["pooling_method"]

            for key in ["projection_dim", "topk_tokens", "use_mrl", "mrl_dims"]:
                if key in vembed_config:
                    model_kwargs[key] = vembed_config[key]

            logger.info("Loaded vembed config: %s", model_kwargs)

        self.model = VisualRetrievalModel(
            model_path,
            encoder_mode=encoder_mode,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        if encoder_mode == "composed":
            self.text_processor = self._load_text_processor(model_path, text_model_name)
            self.image_processor = self._load_image_processor(model_path, image_model_name)
            self.processor = None
        else:
            self.processor = self._load_unified_processor(model_path)
            self.text_processor = None
            self.image_processor = None

    @staticmethod
    def _load_text_processor(model_path: str, fallback_name: str | None):
        try:
            return AutoTokenizer.from_pretrained(model_path)
        except (OSError, ValueError) as exc:
            if not fallback_name:
                raise ValueError(
                    f"Cannot load tokenizer from {model_path} and no text_model_name provided"
                ) from exc
            logger.warning("Tokenizer not in checkpoint, falling back to %s", fallback_name)
            return build_text_processor(fallback_name)

    @staticmethod
    def _load_image_processor(model_path: str, fallback_name: str | None):
        try:
            return AutoImageProcessor.from_pretrained(model_path)
        except (OSError, ValueError) as exc:
            if not fallback_name:
                raise ValueError(
                    f"Cannot load image processor from {model_path} and no image_model_name provided"
                ) from exc
            logger.warning("Image processor not in checkpoint, falling back to %s", fallback_name)
            return build_image_processor(fallback_name)

    @staticmethod
    def _load_unified_processor(model_path: str):
        """Load processor via ProcessorRegistry (auto-detects model type)."""
        return ProcessorRegistry.resolve(model_path)

    def _truncate_and_normalize(self, embeddings: torch.Tensor, normalize: bool) -> np.ndarray:
        if self.mrl_dim:
            embeddings = embeddings[:, : self.mrl_dim]
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode_text(self, text: str | list[str], normalize: bool = True) -> np.ndarray:
        if isinstance(text, str):
            text = [text]

        tokenizer = self.processor or self.text_processor
        with torch.no_grad():
            inputs = tokenizer(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            embeddings = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        return self._truncate_and_normalize(embeddings, normalize)

    def encode_image(
        self,
        image: str | Image.Image | list[str | Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        if not isinstance(image, list):
            image = [image]

        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
            for img in image
        ]

        vision_processor = self.processor or self.image_processor
        with torch.no_grad():
            inputs = vision_processor(
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            embeddings = self.model(pixel_values=inputs["pixel_values"])
        return self._truncate_and_normalize(embeddings, normalize)

    def encode(self, inputs, is_image: bool = False, normalize: bool = True):
        if is_image:
            return self.encode_image(inputs, normalize)
        return self.encode_text(inputs, normalize)
