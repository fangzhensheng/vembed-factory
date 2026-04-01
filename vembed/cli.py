"""vembed-factory CLI entrypoint."""

import argparse
import contextlib
import dataclasses
import logging
import os
import shlex
import signal
import subprocess
import sys

import yaml
from transformers import HfArgumentParser

from vembed.config import (
    apply_false_booleans,
    config_dict_to_argv,
    load_base_config,
    merge_configs,
    parse_override_args,
)
from vembed.hparams import DataArguments, ModelArguments, TrainingArguments
from vembed.validators import validate_config

logger = logging.getLogger(__name__)


def print_usage():
    """Print CLI usage information."""
    usage_text = """
vembed-factory: Multi-modal Embedding Training Framework

Usage:
  vembed train <config.yaml> [options]      Train using a YAML configuration
  vembed list-datasets                      List available datasets
  vembed list-configs                       List available training configurations
  vembed show-dataset <name>                Show details for a dataset
  vembed validate-data <file>               Validate data file format

Examples:
  # Train with a built-in config
  vembed train examples/models/clip/base.yaml

  # Train with parameter overrides
  vembed train examples/models/clip/base.yaml --batch-size 64 --learning-rate 5e-5

  # List available configurations
  vembed list-configs

  # Check a dataset
  vembed show-dataset flickr30k_t2i

  # Dry run (generate config without training)
  vembed train config.yaml --dry-run

For more help, see: https://github.com/your-repo/vembed-factory
    """
    print(usage_text)


def main(args_list=None):
    """CLI entrypoint for vembed-factory with subcommands."""
    # ── Check for subcommands first ───────────────────────────────────
    if args_list is None:
        args_list = sys.argv[1:]

    # Handle help/info subcommands
    if args_list and args_list[0] in ("--help", "-h", "help"):
        print_usage()
        return 0

    if args_list and args_list[0] == "list-datasets":
        from vembed.cli_commands import list_datasets_command
        return list_datasets_command()

    if args_list and args_list[0] == "list-configs":
        from vembed.cli_commands import list_configs_command
        return list_configs_command()

    if args_list and args_list[0] == "show-dataset":
        from vembed.cli_commands import show_dataset_command
        dataset_name = args_list[1] if len(args_list) > 1 else None
        if not dataset_name:
            print("Usage: vembed show-dataset <dataset_name>")
            return 1
        return show_dataset_command(dataset_name)

    # Handle validate-data subcommand
    if args_list and args_list[0] == "validate-data":
        import argparse as ap

        from vembed.entrypoints.validate_data import validate_data_command

        parser = ap.ArgumentParser(prog="vembed validate-data")
        parser.add_argument("data_path", help="Path to data file")
        parser.add_argument("--sample", type=int, default=100)
        parser.add_argument("--column-mapping", nargs="+", default=[])
        parser.add_argument("--check-images", action="store_true")
        parser.add_argument("--image-root", default="")

        args = parser.parse_args(args_list[1:])
        return validate_data_command(args)

    if args_list and args_list[0] == "train":
        args_list = args_list[1:]

    # ── Default: training mode ──────────────────────────────────────────
    # ── 1. Pre-parse: config_file, config_override ──────────────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("config_file", nargs="?", default=None, help="Path to YAML config file")
    pre_parser.add_argument("--config_override", nargs="*", default=[])
    pre_parser.add_argument(
        "--debug_gpu_memory",
        type=int,
        default=None,
        help="Limit GPU memory (GB) for debugging OOM issues",
    )

    known_args, remaining_args = pre_parser.parse_known_args(args_list)

    if known_args.debug_gpu_memory:
        import torch

        gpu_memory_gb = known_args.debug_gpu_memory
        try:
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                device_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                fraction = min(gpu_memory_gb / device_total_gb, 1.0)
                torch.cuda.set_per_process_memory_fraction(fraction)
                logger.info(
                    f"GPU memory limited to {gpu_memory_gb}GB ({fraction:.2%} of {device_total_gb:.1f}GB device)"
                )
            else:
                logger.warning("torch.cuda.set_per_process_memory_fraction not available")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory limit: {e}")

    # ── 2. Load config sources ────────────────────────────────────────
    defaults = load_base_config()
    yaml_config: dict = {}

    # Load YAML config if provided
    if known_args.config_file and known_args.config_file.endswith((".yaml", ".yml")):
        if os.path.exists(known_args.config_file):
            logger.info("Loading config from file: %s", known_args.config_file)
            with open(known_args.config_file) as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            logger.error("Config file not found: %s", known_args.config_file)
            sys.exit(1)

    overrides = parse_override_args(known_args.config_override)

    merged = merge_configs(defaults, {}, yaml_config, overrides)

    # ── 4. Build final argv and parse via HfArgumentParser ────────────
    config_argv = config_dict_to_argv(merged)
    full_argv = config_argv + remaining_args  # CLI args last → override

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=full_argv)
    apply_false_booleans(merged, (model_args, data_args, training_args), remaining_args)

    # ── 5. Build train_config dict from dataclasses ───────────────────
    train_config: dict = {}
    train_config.update(dataclasses.asdict(model_args))
    train_config.update(dataclasses.asdict(data_args))
    train_config.update(dataclasses.asdict(training_args))

    # ── 6. Validate configuration ────────────────────────────────────
    if not validate_config(train_config):
        sys.exit(1)

    # ── 7. Post-processing ───────────────────────────────────────────
    # Alias: lr → learning_rate
    if train_config.get("lr") is not None:
        train_config["learning_rate"] = train_config["lr"]

    # ── Pooling Method Configuration ────────────────────────────────
    # ColBERT needs token-level embeddings, others need global pooling
    loss_type = train_config.get("loss_type")
    pooling_method = train_config.get("pooling_method")

    if loss_type == "colbert":
        if pooling_method and pooling_method != "none":
            logger.warning(
                f"loss_type=colbert requires pooling_method=none. "
                f"Overriding pooling_method={pooling_method} with none."
            )
        train_config["pooling_method"] = "none"
    else:
        # Non-ColBERT: "none" is invalid, must use global pooling
        if pooling_method == "none":
            logger.warning(
                f"loss_type={loss_type} requires global pooling. "
                f"Resetting pooling_method=none to cls."
            )
            train_config["pooling_method"] = "cls"
        elif pooling_method is None:
            train_config["pooling_method"] = "cls"

    # model_name_or_path → model_name fallback
    if train_config.get("model_name_or_path") and not train_config.get("model_name"):
        train_config["model_name"] = train_config["model_name_or_path"]

    # FSDP + gradient cache compatibility notice
    if train_config.get("use_fsdp") and train_config.get("use_gradient_cache"):
        logger.info(
            "Using FSDP with Gradient Cache. "
            "This enables memory-efficient training for very large models. "
            "Ensure all GPUs have consistent FSDP wrapping."
        )

    # ── 8. Save config & launch training ──────────────────────────────
    os.makedirs(train_config["output_dir"], exist_ok=True)
    config_path = os.path.join(train_config["output_dir"], ".train_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

    # Build launch command
    launch_parts = ["accelerate", "launch"]

    if train_config.get("use_fsdp"):
        root_configs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        fsdp_config = os.path.join(root_configs, "accelerate_fsdp.yaml")
        if not os.path.exists(fsdp_config):
            fsdp_config = os.path.join(os.path.dirname(__file__), "configs", "accelerate_fsdp.yaml")
        launch_parts.extend(["--config_file", fsdp_config])

    if train_config.get("num_gpus"):
        launch_parts.extend(["--num_processes", str(train_config["num_gpus"])])

    script_path = os.path.join(os.path.dirname(__file__), "entrypoints", "train.py")
    if not os.path.exists(script_path):
        logger.error("Training script not found at %s", script_path)
        sys.exit(1)

    launch_parts.extend([script_path, "--config", config_path])
    logger.info("Running command: %s", " ".join(shlex.quote(p) for p in launch_parts))

    if train_config.get("dry_run"):
        logger.info("Dry run enabled. Exiting before launch.")
        return 0

    try:
        process = subprocess.Popen(launch_parts, start_new_session=True)
        process.wait()
        exit_code = process.returncode
    except KeyboardInterrupt:
        logger.warning("Interrupted — killing training process group...")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        exit_code = 1

    if exit_code == 0:
        logger.info("Training complete. Model saved to: %s", train_config["output_dir"])
    else:
        logger.error("Training failed (exit code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
