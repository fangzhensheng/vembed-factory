"""Checkpoint management for training."""

import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str,
    model: Any,
    accelerator: Accelerator,
    processor: Any = None,
    config: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with model, processor, and vembed config.

    Only the global main process (rank 0) saves to prevent multi-node file conflicts.
    All ranks synchronize after save completes.

    Args:
        path: Directory path to save checkpoint.
        model: The model to save.
        accelerator: Accelerate instance for distributed saving.
        processor: Optional processor to save.
        config: Optional configuration dict for vembed-specific config.
        training_state: Optional training state dict with global_step, epoch, best_metric, patience_counter.

    Note:
        CRITICAL for multi-node safety: Uses is_main_process (not is_local_main_process)
        to ensure only rank 0 globally saves. Without this, multiple nodes' rank-0
        processes could write simultaneously to shared storage, causing corruption.
    """
    if not accelerator.is_main_process:
        # Wait for main rank (rank 0) to finish saving before continuing
        accelerator.wait_for_everyone()
        return

    accelerator.save_state(path)
    accelerator.unwrap_model(model).save_pretrained(path)
    if processor:
        processor.save_pretrained(path)

    # Persist vembed-specific config (topk_tokens, pooling, etc.)
    if config:
        _save_vembed_config(path, config)

    # Persist training state for resuming
    if training_state is not None:
        _save_training_state(path, training_state)

    # CRITICAL: Synchronize all ranks after checkpoint save completes.
    # Prevents non-main processes from resuming training while main process
    # is still writing to disk.
    accelerator.wait_for_everyone()


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


def _save_training_state(path: str, training_state: dict[str, Any]) -> None:
    """Save training state (global_step, epoch, metrics, etc.) to JSON file.

    Args:
        path: Directory path to save training state.
        training_state: Dict with keys: global_step, epoch, best_metric, patience_counter.
    """
    state_path = os.path.join(path, "training_state.json")
    with open(state_path, "w") as fp:
        json.dump(training_state, fp, indent=2)
    logger.info(f"Saved training_state.json → {state_path}")


def load_checkpoint(
    path: str,
    model: Any,
    accelerator: Accelerator,
    optimizer: Any = None,
    scheduler: Any = None,
    processor: Any = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Load checkpoint and restore training state.

    Args:
        path: Path to checkpoint directory.
        model: Model to load weights into.
        accelerator: Accelerator instance.
        optimizer: Optimizer to load state into (required if mode='full').
        scheduler: LR scheduler to load state into (required if mode='full').
        processor: Processor to load (optional).
        mode: "full" (restore optimizer/scheduler/RNG) or "model_only" (just weights).

    Returns:
        Dictionary with training state: {
            'global_step': int,
            'epoch': int,
            'best_metric': float,
            'patience_counter': int,
        }

    Raises:
        FileNotFoundError: If checkpoint directory does not exist.
        ValueError: If mode is invalid.
    """
    if mode not in ("full", "model_only"):
        raise ValueError(f"Invalid resume mode: {mode}. Must be 'full' or 'model_only'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_path = Path(path)
    logger.info(f"Loading checkpoint from {path} (mode={mode})")

    # Load model weights
    model_path = checkpoint_path / "pytorch_model.bin"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=accelerator.device)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model weights from {model_path}")
    else:
        logger.warning(f"Model weights file not found at {model_path}")

    # Load processor if exists
    if processor is not None:
        try:
            processor.load_pretrained(str(checkpoint_path))
            logger.info(f"Loaded processor from {checkpoint_path}")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Failed to load processor: {e}")

    # Load optimizer + scheduler state (full mode)
    if mode == "full":
        try:
            accelerator.load_state(checkpoint_path)
            logger.info(f"Loaded optimizer and scheduler state from {checkpoint_path}")
        except RuntimeError as e:
            logger.warning(f"Failed to load optimizer/scheduler state: {e}")

    # Load training state
    training_state = {}
    state_file = checkpoint_path / "training_state.json"
    if state_file.exists():
        with open(state_file) as f:
            training_state = json.load(f)
        logger.info(
            f"Loaded training state: global_step={training_state.get('global_step')}, epoch={training_state.get('epoch')}"
        )
    else:
        logger.warning(f"Training state file not found at {state_file}")

    return training_state
