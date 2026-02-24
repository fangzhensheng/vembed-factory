from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel

from ..base import BaseEmbeddingModel, pool, resolve_pretrained_kwargs
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


def _unwrap_clip_model(backbone: torch.nn.Module) -> torch.nn.Module | None:
    """Return the underlying CLIP-like model, unwrapping DDP / PeftModel / LoRA if needed.

    When ``peft.get_peft_model`` wraps a CLIPModel, the wrapper's
    ``__getattr__`` delegation may fail for ``get_text_features`` /
    ``get_image_features`` depending on the peft version.  This helper
    walks through the wrapping layers to find the original model that
    exposes those methods.

    LoRA injects adapter weights *in-place* into the original model's
    linear layers, so calling ``get_text_features`` on the unwrapped model
    still uses the LoRA-adapted weights.
    """

    def _has_clip_methods(m: torch.nn.Module) -> bool:
        return hasattr(m, "get_text_features") and hasattr(m, "get_image_features")

    # Strip DDP / FSDP wrapper first (.module is the unwrapped model)
    if hasattr(backbone, "module"):
        backbone = backbone.module

    # Direct check — no wrapping or a peft version that delegates correctly
    if _has_clip_methods(backbone):
        return backbone

    # PeftModel.get_base_model() → original model (with LoRA layers injected)
    if hasattr(backbone, "get_base_model"):
        try:
            base = backbone.get_base_model()
            if _has_clip_methods(base):
                return base
        except Exception:
            pass

    # Fallback: walk common wrapping attributes used by peft / LoraModel
    for attr in ("model", "base_model"):
        candidate = getattr(backbone, attr, None)
        if candidate is None:
            continue
        if _has_clip_methods(candidate):
            return candidate
        # LoraModel stores the original model under .model
        inner = getattr(candidate, "model", None)
        if inner is not None and _has_clip_methods(inner):
            return inner

    return None


def _ensure_tensor(
    output: object,
    attention_mask: torch.Tensor | None = None,
    pooling_method: str = "mean",
) -> torch.Tensor:
    """Guarantee a model output is a plain Tensor.

    CLIP ``get_text_features`` / ``get_image_features`` normally return
    Tensors, but some PeftModel wrapping configurations return a
    HuggingFace ``ModelOutput`` instead.  This converts any non-Tensor
    output via :func:`pool`.
    """
    if isinstance(output, torch.Tensor):
        return output
    # ModelOutput fallback — extract via pooling
    return pool(output, attention_mask, pooling_method)


@ModelRegistry.register("auto")
class AutoEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_name = config["model_name_or_path"]
        self.pooling_method = (
            config.get("pooling_method")
            or config.get("pooling_strategy")
            or config.get("pooling")
            or "cls"
        )
        self.use_mrl = config.get("use_mrl", False)

        # Attention-guided top-K token pruning for late-interaction (ColBERT) models.
        # topk_tokens > 0: keep only the K most relevant patches (CLS-patch cosine).
        # topk_tokens = 0: keep all tokens (full token interaction, default).
        self.topk_tokens = int(config.get("topk_tokens", 0))

        self.backbone = AutoModel.from_pretrained(
            self.model_name,
            **resolve_pretrained_kwargs(config),
        )

        # Automatically load LoRA adapter if present in the model directory
        adapter_config_path = os.path.join(self.model_name, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                from peft import PeftModel

                logger.info(f"Loading LoRA adapter from {self.model_name}")
                self.backbone = PeftModel.from_pretrained(self.backbone, self.model_name)
                # Merge for efficiency during inference/embedding generation
                self.backbone = self.backbone.merge_and_unload()
            except ImportError:
                logger.warning("Found adapter_config.json but 'peft' is not installed.")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}")

        self.hf_config = self.backbone.config

        hidden_size = (
            getattr(self.hf_config, "hidden_size", None)
            or getattr(self.hf_config, "projection_dim", None)
            or getattr(getattr(self.hf_config, "text_config", None), "hidden_size", 768)
        )

        self.mrl_dims = config.get("mrl_dims") or [hidden_size]
        if self.use_mrl:
            self.mrl_dims = sorted(self.mrl_dims, reverse=True)

        self.projection_dim = config.get("projection_dim")
        self.projection_head = None
        if self.projection_dim and self.projection_dim != hidden_size:
            self.projection_head = torch.nn.Linear(hidden_size, self.projection_dim, bias=False)
            # Try loading existing projection head
            proj_path = os.path.join(self.model_name, "projection_head.pt")
            if os.path.isfile(proj_path):
                logger.info(f"Loading projection head from {proj_path}")
                self.projection_head.load_state_dict(torch.load(proj_path, map_location="cpu"))

    def save_pretrained(self, save_directory: str):
        """Save the model weights (and projection head) to a directory."""
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_directory)
        else:
            # Fallback for plain nn.Module
            torch.save(
                self.backbone.state_dict(), os.path.join(save_directory, "pytorch_model.bin")
            )
            if hasattr(self.backbone, "config") and hasattr(
                self.backbone.config, "save_pretrained"
            ):
                self.backbone.config.save_pretrained(save_directory)

        if self.projection_head is not None:
            torch.save(
                self.projection_head.state_dict(),
                os.path.join(save_directory, "projection_head.pt"),
            )

    def _attention_topk_tokens(self, hidden: torch.Tensor, topk: int) -> torch.Tensor:
        """Select top-K patches by CLS-patch relevance (content-aware pruning).

        Uses cosine similarity between the [CLS] token and each patch token
        as an importance score.  This closely approximates the CLS self-attention
        map from the last transformer layer while requiring **zero** extra memory
        (no ``output_attentions=True`` needed).

        Key properties:
          * Preferentially keeps subject patches, filters out background noise.
          * Preserves fine-grained detail on the object of interest.
          * The selection is differentiable (``torch.gather`` supports autograd).

        Args:
            hidden: ``[B, 1+N, D]`` — CLS token followed by N patch tokens.
            topk:   Number of patches to keep.

        Returns:
            ``[B, 1+topk, D]`` — CLS token followed by the top-K patches,
            sorted by original spatial position to maintain rough locality.
        """
        B, L, D = hidden.shape
        cls_token = hidden[:, :1, :]  # [B, 1, D]
        patches = hidden[:, 1:, :]  # [B, N, D]

        n_patches = patches.shape[1]
        if topk <= 0 or topk >= n_patches:
            return hidden

        # CLS-patch importance: cosine similarity [B, N]
        scores = torch.bmm(
            F.normalize(patches, p=2, dim=-1),
            F.normalize(cls_token, p=2, dim=-1).transpose(1, 2),
        ).squeeze(-1)

        # Select top-K, then sort indices to preserve spatial order
        topk_indices = scores.topk(topk, dim=1).indices  # [B, K]
        topk_indices = topk_indices.sort(dim=1).values  # [B, K]

        # Gather the selected patches
        selected = patches.gather(
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, K, D]

        return torch.cat([cls_token, selected], dim=1)  # [B, 1+K, D]

    def _select_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply attention-guided top-K pruning if ``topk_tokens > 0``."""
        if self.topk_tokens > 0:
            return self._attention_topk_tokens(hidden, self.topk_tokens)
        return hidden

    def _project(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.projection_head is not None:
            return self.projection_head(embeddings)
        return embeddings

    def _find_clip_model(self) -> torch.nn.Module | None:
        """Locate the CLIP-like model, handling PeftModel wrapping."""
        return _unwrap_clip_model(self.backbone)

    @property
    def _is_clip_like(self) -> bool:
        return self._find_clip_model() is not None

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        clip_model = self._find_clip_model()
        if clip_model is not None:
            if input_ids is not None and pixel_values is None:
                out = clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )
                return self._project(_ensure_tensor(out, attention_mask, self.pooling_method))
            if pixel_values is not None and input_ids is None:
                out = clip_model.get_image_features(pixel_values=pixel_values, **kwargs)
                return self._project(_ensure_tensor(out, None, self.pooling_method))
            if input_ids is not None and pixel_values is not None:
                return (
                    self._project(
                        _ensure_tensor(
                            clip_model.get_text_features(
                                input_ids=input_ids, attention_mask=attention_mask
                            ),
                            attention_mask,
                            self.pooling_method,
                        )
                    ),
                    self._project(
                        _ensure_tensor(
                            clip_model.get_image_features(pixel_values=pixel_values),
                            None,
                            self.pooling_method,
                        )
                    ),
                )

        # Multimodal pass (e.g. Qwen-VL)
        if input_ids is not None and pixel_values is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs,
            )
            return self._project(pool(outputs, attention_mask, self.pooling_method))

        if input_ids is not None:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return self._project(pool(outputs, attention_mask, self.pooling_method))

        if pixel_values is not None:
            outputs = self.backbone(pixel_values=pixel_values, **kwargs)
            emb = pool(outputs, attention_mask=None, method=self.pooling_method)
            if self.pooling_method == "none":
                emb = self._select_tokens(emb)
            return self._project(emb)

    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        clip_model = self._find_clip_model()
        if clip_model is not None:
            out = clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            return self._project(_ensure_tensor(out, attention_mask, self.pooling_method))
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_image_features(self, pixel_values=None, **kwargs):
        clip_model = self._find_clip_model()
        if clip_model is not None:
            out = clip_model.get_image_features(pixel_values=pixel_values, **kwargs)
            return self._project(_ensure_tensor(out, None, self.pooling_method))
        return self.forward(pixel_values=pixel_values, **kwargs)
