from typing import Any

import torch.nn as nn
from transformers import PreTrainedModel

from .registry import ModelRegistry


class VisualRetrievalModel(nn.Module):
    """Facade that delegates to a registered backend based on encoder_mode."""

    def __init__(
        self,
        model_name_or_path: str,
        pooling_method: str | None = None,
        use_mrl: bool = False,
        mrl_dims: list[int] | None = None,
        encoder_mode: str = "auto",
        text_model_name: str | None = None,
        image_model_name: str | None = None,
        attn_implementation: str | None = None,
        torch_dtype: str | None = None,
        projection_dim: int | None = None,
        topk_tokens: int = 0,
    ):
        super().__init__()

        self.config_dict = {
            "model_name_or_path": model_name_or_path,
            "pooling_method": pooling_method,
            "use_mrl": use_mrl,
            "mrl_dims": mrl_dims,
            "encoder_mode": encoder_mode,
            "text_model_name": text_model_name,
            "image_model_name": image_model_name,
            "attn_implementation": attn_implementation,
            "torch_dtype": torch_dtype,
            "projection_dim": projection_dim,
            "topk_tokens": topk_tokens,
        }

        backend_name = self._resolve_backend(encoder_mode, text_model_name, image_model_name)
        backend_cls = ModelRegistry.get(backend_name)
        self.backend = backend_cls(self.config_dict)

        self.mrl_dims = getattr(self.backend, "mrl_dims", None)
        self.config = getattr(self.backend, "hf_config", None)

    @staticmethod
    def _resolve_backend(
        encoder_mode: str,
        text_model_name: str | None,
        image_model_name: str | None,
    ) -> str:
        if encoder_mode == "composed" or (text_model_name and image_model_name):
            return "composed"

        # Check if the requested mode is explicitly registered
        if encoder_mode != "auto" and encoder_mode in ModelRegistry.list_models():
            return encoder_mode

        return "auto"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend(*args, **kwargs)

    def save_pretrained(self, save_directory: str) -> None:
        if hasattr(self.backend, "save_pretrained") or isinstance(self.backend, PreTrainedModel):
            self.backend.save_pretrained(save_directory)
        else:
            # Fallback for custom models without save_pretrained
            # This might need torch.save, but sticking to interface for now
            pass

    def get_text_features(self, *args: Any, **kwargs: Any) -> Any:
        backend: Any = self.backend
        if hasattr(backend, "get_text_features"):
            return backend.get_text_features(*args, **kwargs)

        if "input_ids" in kwargs:
            return self.forward(**kwargs)

        raise AttributeError("Backend does not support get_text_features")

    def get_image_features(self, *args: Any, **kwargs: Any) -> Any:
        backend: Any = self.backend
        if hasattr(backend, "get_image_features"):
            return backend.get_image_features(*args, **kwargs)

        if "pixel_values" in kwargs:
            return self.forward(**kwargs)

        raise AttributeError("Backend does not support get_image_features")
