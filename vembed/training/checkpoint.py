"""Checkpoint management for training."""

import json
import os
from typing import Any

from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str,
    model: Any,
    accelerator: Accelerator,
    processor: Any = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with model, processor, and vembed config.

    Args:
        path: Directory path to save checkpoint.
        model: The model to save.
        accelerator: Accelerate instance for distributed saving.
        processor: Optional processor to save.
        config: Optional configuration dict for vembed-specific config.

    Note:
        Only the main process saves the checkpoint to avoid conflicts.
    """
    if not accelerator.is_local_main_process:
        return

    accelerator.save_state(path)
    accelerator.unwrap_model(model).save_pretrained(path)
    if processor:
        processor.save_pretrained(path)

    # Persist vembed-specific config (topk_tokens, pooling, etc.)
    if config:
        _save_vembed_config(path, config)


def _save_vembed_config(path: str, config: dict[str, Any]) -> None:
    """Save vembed-specific configuration to JSON file.

    Args:
        path: Directory path to save config.
        config: Full configuration dict.
    """
    vembed_cfg = {
        "pooling_method": config.get("pooling_method"),
        "projection_dim": config.get("projection_dim"),
        "topk_tokens": int(config.get("topk_tokens", 0)),
        "retrieval_mode": config.get("retrieval_mode", "t2i"),
        "loss_type": config.get("loss_type", "infonce"),
        "use_mrl": config.get("use_mrl", False),
        "mrl_dims": config.get("mrl_dims"),
        "encoder_mode": config.get("encoder_mode", "auto"),
        "text_model_name": config.get("text_model_name"),
        "image_model_name": config.get("image_model_name"),
    }
    cfg_path = os.path.join(path, "vembed_config.json")
    with open(cfg_path, "w") as fp:
        json.dump(vembed_cfg, fp, indent=2)
    logger.info(f"Saved vembed_config.json → {cfg_path}")
