import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..base import BaseEmbeddingModel, _extract_hidden_state, resolve_pretrained_kwargs, disable_kv_cache
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("qwen3_vl")
class Qwen3EmbeddingModel(BaseEmbeddingModel):
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
        text_cfg = getattr(self.hf_config, "text_config", None)
        if text_cfg:
            self.feature_dim = text_cfg.hidden_size
        else:
            self.feature_dim = self.hf_config.hidden_size

        mrl_dims = config.get("mrl_dims")
        self.mrl_dims = mrl_dims or [self.feature_dim]
        if self.use_mrl:
            self.mrl_dims = sorted(self.mrl_dims, reverse=True)

        # Automatically load LoRA adapter if present in the model directory
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

    @staticmethod
    def _probe_hidden_size(model_name: str) -> int:
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            text_cfg = getattr(cfg, "text_config", None)
            return text_cfg.hidden_size if text_cfg else cfg.hidden_size
        except Exception:
            return 1536

    @staticmethod
    def _pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last hidden state by attention mask for embeddings.

        This aligns with Qwen3-VL official implementation:
        1. Flip attention mask
        2. Find first '1' (which corresponds to the last '1' in original)
        3. Extract hidden state at that position
        """
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        # Calculate original index: seq_len - 1 - index_in_flipped
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row, col]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden = _extract_hidden_state(outputs)
        embeddings = self._pool_last_token(hidden, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)
