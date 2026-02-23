"""vembed-factory CLI entrypoint.

Parses configuration from defaults, presets, YAML files, and CLI overrides,
then launches training via ``accelerate launch``.
"""

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
    PRESETS,
    apply_false_booleans,
    config_dict_to_argv,
    infer_preset_from_model_name,
    load_base_config,
    merge_configs,
    parse_override_args,
)
from vembed.hparams import DataArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)


def main(args_list=None):
    # ── 1. Pre-parse: preset, config_file, config_override ────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("config_file", nargs="?", default=None, help="Path to YAML config file")
    pre_parser.add_argument("--preset", type=str, default="custom")
    pre_parser.add_argument("--model_type", type=str, default=None)  # deprecated alias
    pre_parser.add_argument("--config_override", nargs="*", default=[])

    known_args, remaining_args = pre_parser.parse_known_args(args_list)

    # Handle deprecated --model_type
    if known_args.model_type:
        logger.warning("'--model_type' is deprecated. Use '--preset' instead.")
        if known_args.preset == "custom":
            known_args.preset = known_args.model_type

    # ── 2. Resolve config sources ─────────────────────────────────────
    yaml_config: dict = {}
    if known_args.config_file:
        if known_args.config_file.endswith((".yaml", ".yml")) and os.path.exists(
            known_args.config_file
        ):
            logger.info("Loading config from file: %s", known_args.config_file)
            with open(known_args.config_file) as f:
                yaml_config = yaml.safe_load(f) or {}
        elif known_args.config_file in PRESETS:
            known_args.preset = known_args.config_file

    preset_name = known_args.preset
    defaults = load_base_config()
    preset_config = dict(PRESETS.get(preset_name, {}))

    if preset_name != "custom":
        logger.info("Using preset: %s", preset_name.upper())

    # ── 3. Infer preset from model_name if needed ─────────────────────
    if preset_name == "custom" and not yaml_config and "--model_name" in remaining_args:
        idx = remaining_args.index("--model_name") + 1
        if idx < len(remaining_args):
            preset_config.update(infer_preset_from_model_name(remaining_args[idx]))

    overrides = parse_override_args(known_args.config_override)
    merged = merge_configs(defaults, preset_config, yaml_config, overrides)

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

    # ── 6. Post-processing & validation ───────────────────────────────
    # Alias: lr → learning_rate
    if train_config.get("lr") is not None:
        train_config["learning_rate"] = train_config["lr"]

    train_config["pooling_strategy"] = (
        "none" if train_config.get("loss_type") == "colbert" else "last"
    )

    # Composed mode validation
    if train_config.get("encoder_mode") == "composed" and not (
        train_config.get("text_model_name") and train_config.get("image_model_name")
    ):
        logger.error("Composed mode requires --text_model_name and --image_model_name")
        sys.exit(1)

    # model_name_or_path → model_name fallback
    if train_config.get("model_name_or_path") and not train_config.get("model_name"):
        train_config["model_name"] = train_config["model_name_or_path"]

    # FSDP + gradient cache incompatibility
    if train_config.get("use_fsdp") and train_config.get("use_gradient_cache"):
        logger.warning("Disabling Gradient Cache (incompatible with FSDP)")
        train_config["use_gradient_cache"] = False

    # ── 7. Save config & launch training ──────────────────────────────
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
