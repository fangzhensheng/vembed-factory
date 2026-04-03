"""Configuration management for training.

Handles configuration loading, parsing, merging, and validation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from vembed.config import load_base_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="vembed-factory training script")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (auto-set by accelerate)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    return parser.parse_args()


def _load_dataset_info(dataset_name: str) -> dict[str, Any] | None:
    """Load dataset configuration from dataset_info.json.

    Args:
        dataset_name: Name of dataset to load.

    Returns:
        Dataset configuration dict or None if not found.
    """
    try:
        # Search for dataset_info.json in multiple possible locations
        # Priority: current dir, vembed package dir, repo root/examples
        search_paths = [
            Path.cwd() / "examples" / "dataset_info.json",  # Current working dir
            Path(__file__).parent.parent.parent / "examples" / "dataset_info.json",  # Repo root
            Path(__file__).parent / "dataset_info.json",  # Fallback in vembed dir
        ]

        dataset_file = None
        for path in search_paths:
            if path.exists():
                dataset_file = path
                break

        if not dataset_file:
            logger.warning(
                "dataset_info.json not found in any of: %s",
                ", ".join(str(p) for p in search_paths),
            )
            return None

        with open(dataset_file) as f:
            dataset_info = json.load(f)

        return dataset_info.get(dataset_name)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to load dataset_info for '%s': %s", dataset_name, e)
        return None


def inject_dataset_info(config: dict[str, Any]) -> None:
    """Inject dataset paths from dataset_info.json into the configuration dict in-place."""
    if not config.get("dataset_name"):
        return

    dataset_name = config["dataset_name"]
    dataset_info = _load_dataset_info(dataset_name)
    if not dataset_info:
        logger.warning("Dataset '%s' not found in dataset_info.json", dataset_name)
        return

    if not config.get("data_path") or config.get("data_path") == "data/train.jsonl":
        config["data_path"] = dataset_info.get("file_name")
    if not config.get("val_data_path"):
        config["val_data_path"] = dataset_info.get("val_file_name")
    if not config.get("image_root") and dataset_info.get("image_root"):
        config["image_root"] = dataset_info.get("image_root")
    if not config.get("column_mapping") and dataset_info.get("columns"):
        config["column_mapping"] = dataset_info.get("columns")

    logger.info(
        "Loaded dataset '%s': data_path=%s, val_data_path=%s",
        dataset_name,
        config.get("data_path"),
        config.get("val_data_path"),
    )


def load_and_parse_config() -> dict[str, Any]:
    """Load and parse configuration from args and files.

    Returns:
        Merged configuration dictionary with the following hierarchy:
        1. Base config (defaults)
        2. File config (if --config provided)
        3. Dataset info (if dataset_name specified in YAML)
        4. Gradient checkpointing flag (if --gradient_checkpointing)

    Raises:
        SystemExit: If required config values are missing.
    """
    args, unknown_args = argparse.ArgumentParser(allow_abbrev=False).parse_known_args()

    # We parse the known args manually since we removed parse_args() strict check
    # to allow arbitrary kwargs to pass through to HfArgumentParser later
    parser = argparse.ArgumentParser(description="vembed-factory training script", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    args, _ = parser.parse_known_args()

    # Load base configuration
    config = load_base_config()

    # Merge file configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)

    # Load dataset_info if dataset_name is specified
    inject_dataset_info(config)

    # Apply gradient checkpointing flag if provided
    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True

    return config


def prepare_output_dir(config: dict[str, Any]) -> None:
    """Create output directory if it doesn't exist.

    Args:
        config: Configuration dictionary containing 'output_dir'.
    """
    os.makedirs(config["output_dir"], exist_ok=True)


def get_distributed_config(config: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Extract distributed training configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of:
        - use_grad_checkpointing: Whether to use gradient checkpointing
        - use_gradient_cache: Whether to use gradient cache
        - find_unused: Whether to find unused parameters in DDP

    Note:
        For gradient checkpointing: find_unused_parameters is automatically set to False.

        For gradient cache with dual-encoder models (CLIP, SigLIP):
        find_unused_parameters should be True because each chunk may only use one encoder
        (text or image), leaving the other encoder's parameters unused in that chunk.
        Set ddp_find_unused_parameters: true in your config for dual-encoder models.

        UPDATE: Gradient cache now adds zero gradients to unused params internally,
        allowing find_unused_parameters=False (better performance) by default.
        Users can still override by explicitly setting ddp_find_unused_parameters.
    """
    use_grad_checkpointing = config.get("gradient_checkpointing", False)
    use_gradient_cache = config.get("use_gradient_cache", False)

    # Check if user explicitly set ddp_find_unused_parameters
    user_set_find_unused = "ddp_find_unused_parameters" in config

    # User explicitly set the value, use it; otherwise default to False for better performance
    # Gradient cache now handles DDP compatibility by adding zero gradients
    find_unused = bool(config.get("ddp_find_unused_parameters")) if user_set_find_unused else False

    # Disable find_unused_parameters when using gradient checkpointing
    # (overrides user setting if using gradient checkpointing)
    if use_grad_checkpointing:
        find_unused = False

    return use_grad_checkpointing, use_gradient_cache, find_unused
