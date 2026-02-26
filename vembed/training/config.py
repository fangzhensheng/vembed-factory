"""Configuration management for training.

Handles configuration loading, parsing, merging, and validation.
"""

import argparse
import os
from typing import Any

import yaml

from vembed.config import load_base_config, parse_override_args


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
        "--config_override",
        type=str,
        nargs="*",
        help="Override config keys, e.g., model_name=bert batch_size=32",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    return parser.parse_args()


def load_and_parse_config() -> dict[str, Any]:
    """Load and parse configuration from args and files.

    Returns:
        Merged configuration dictionary with the following hierarchy:
        1. Base config (defaults)
        2. File config (if --config provided)
        3. CLI overrides (if --config_override provided)
        4. Gradient checkpointing flag (if --gradient_checkpointing)

    Raises:
        SystemExit: If required config values are missing.
    """
    args = parse_args()

    # Load base configuration
    config = load_base_config()

    # Merge file configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)

    # Apply CLI overrides
    if args.config_override:
        config.update(parse_override_args(args.config_override))

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
        When using gradient checkpointing or gradient cache, find_unused_parameters
        is automatically set to False as these techniques modify parameter usage patterns.
    """
    use_grad_checkpointing = config.get("gradient_checkpointing", False)
    use_gradient_cache = config.get("use_gradient_cache", False)
    find_unused = bool(config.get("ddp_find_unused_parameters", True))

    # Disable find_unused_parameters when using gradient optimization techniques
    if use_grad_checkpointing or use_gradient_cache:
        find_unused = False

    return use_grad_checkpointing, use_gradient_cache, find_unused
