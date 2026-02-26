"""Optimizer and scheduler builders."""

import math
from typing import Any

import torch
from accelerate.logging import get_logger
from transformers import get_scheduler

logger = get_logger(__name__)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build AdamW optimizer with weight decay for non-bias and LayerNorm parameters.

    Args:
        model: The model to optimize.
        config: Configuration dict with 'lr' and optional 'weight_decay'.

    Returns:
        Configured AdamW optimizer.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(param_groups, lr=float(config["lr"]))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    num_epochs: int,
    steps_per_epoch: int,
) -> tuple[Any, int]:
    """Build learning rate scheduler with warmup.

    Args:
        optimizer: The optimizer.
        config: Configuration dict with optional 'scheduler_type', 'warmup_steps', 'warmup_ratio'.
        num_epochs: Number of training epochs.
        steps_per_epoch: Number of steps per epoch.

    Returns:
        Tuple of (scheduler, warmup_steps).
    """
    max_train_steps = num_epochs * steps_per_epoch

    warmup_steps = int(config.get("warmup_steps", 0))
    if warmup_steps == 0:
        warmup_steps = math.ceil(max_train_steps * float(config.get("warmup_ratio", 0.1)))

    scheduler = get_scheduler(
        name=config.get("scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    return scheduler, warmup_steps


def resolve_tracker(report_to: str | None) -> tuple[str | None, dict[str, Any]]:
    """Resolve experiment tracker configuration for accelerate.

    Args:
        report_to: Tracker name (e.g., 'wandb', 'tensorboard', 'swanlab', 'none').

    Returns:
        Tuple of (log_with, init_kwargs) for accelerator.init_trackers().
        Returns (None, {}) if no tracker or tracker not available.
    """
    if report_to in (None, "none"):
        return None, {}

    init_kwargs: dict[str, Any] = {}

    # swanlab became a built-in tracker in accelerate 1.8.0
    if report_to == "swanlab":
        try:
            import accelerate
            from packaging.version import Version

            if Version(accelerate.__version__) >= Version("1.8.0"):
                return "swanlab", {"swanlab": {"experiment_name": "vembed-factory"}}
        except ImportError:
            pass

        try:
            from swanlab.integration.accelerate import SwanLabTracker

            tracker = SwanLabTracker("vembed-factory")
            return tracker, {}
        except ImportError:
            logger.warning(
                "swanlab requested but neither accelerate>=1.8.0 nor swanlab package found. "
                "Install with: pip install swanlab"
            )
            return None, {}

    return report_to, init_kwargs
