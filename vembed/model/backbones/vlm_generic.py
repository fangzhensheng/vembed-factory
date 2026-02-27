import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..base import BaseEmbeddingModel, _extract_hidden_state, resolve_pretrained_kwargs, disable_kv_cache
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("vlm_generic")
class GenericVLMEmbeddingModel(BaseEmbeddingModel):
    """Wrapper for causal VLMs (Qwen-VL, InternVL, Gemma-VL, etc.).

    Uses last non-padding token as the sentence embedding — the standard
    approach for decoder-only models that lack a dedicated [CLS] token.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_name = config["model_name_or_path"]
        self.use_mrl = config.get("use_mrl", False)

        self.feature_dim = self._probe_hidden_size(self.model_name)

        self.backbone = AutoModel.from_pretrained(
            self.model_name,
            **resolve_pretrained_kwargs(config),
        )
        disable_kv_cache(self.backbone)
        self.hf_config = self.backbone.config
        adapter_config_path = os.path.join(self.model_name, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                from peft import PeftModel

                logger.info(f"Loading LoRA adapter from {self.model_name}")
                self.backbone = PeftModel.from_pretrained(self.backbone, self.model_name)
                # Merge for efficiency during inference/embedding generation
                self.backbone = self.backbone.merge_and_unload()
            except ImportError as e:
                logger.warning("peft not installed, skipping LoRA adapter loading: %s", e)

        mrl_dims = config.get("mrl_dims")
        self.mrl_dims = mrl_dims or [self.feature_dim]
        if self.use_mrl:
            self.mrl_dims = sorted(self.mrl_dims, reverse=True)

        self.projection_dim = config.get("projection_dim")
        self.projection_head = None
        if self.projection_dim and self.projection_dim != self.feature_dim:
            self.projection_head = torch.nn.Linear(
                self.feature_dim, self.projection_dim, bias=False
            )
            # Try loading existing projection head
            proj_path = os.path.join(self.model_name, "projection_head.pt")
            if os.path.isfile(proj_path):
                self.projection_head.load_state_dict(torch.load(proj_path, map_location="cpu"))

    @staticmethod
    def _probe_hidden_size(model_name: str) -> int:
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            text_cfg = getattr(cfg, "text_config", None)
            return text_cfg.hidden_size if text_cfg else cfg.hidden_size
        except Exception:
            return 4096

    @staticmethod
    def _pool_last_token(hidden: torch.Tensor, attention_mask) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1]
        # Find the position of the last non-padding token per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, seq_lengths]

    @staticmethod
    def _pool_colbert(hidden: torch.Tensor, _attention_mask) -> torch.Tensor:
        """ColBERT-style: L2-normalize every token independently."""
        return F.normalize(hidden, p=2, dim=2)

    def _project(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.projection_head is not None:
            return self.projection_head(embeddings)
        return embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pooling_strategy="last",
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden = _extract_hidden_state(outputs)

        if pooling_strategy == "colbert":
            return F.normalize(self._project(hidden), p=2, dim=2)

        embeddings = self._pool_last_token(hidden, attention_mask)
        return F.normalize(self._project(embeddings), p=2, dim=1)

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)
        if self.projection_head is not None:
            torch.save(
                self.projection_head.state_dict(),
                os.path.join(save_directory, "projection_head.pt"),
            )
