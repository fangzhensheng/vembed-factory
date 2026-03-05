"""Shared configuration utilities."""

import ast
import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from vembed.hparams import DataArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)

# Cache valid fields across dataclasses
_VALID_FIELDS: set[str] | None = None


def _get_valid_fields() -> set[str]:
    global _VALID_FIELDS
    if _VALID_FIELDS is None:
        _VALID_FIELDS = set()
        for dc in (ModelArguments, DataArguments, TrainingArguments):
            for f in dataclasses.fields(dc):
                _VALID_FIELDS.add(f.name)
    return _VALID_FIELDS


PRESETS: dict[str, Any] = {}


def load_presets() -> dict[str, Any]:
    """Discover and load all YAML preset files from configs/ directory."""
    project_root = Path(__file__).parent.parent
    preset_dir = project_root / "configs"

    if not preset_dir.is_dir():
        preset_dir = Path(__file__).parent / "configs"

    if not preset_dir.is_dir():
        return PRESETS

    for cfg_file in preset_dir.glob("*.y*ml"):
        if cfg_file.name == "defaults.yaml":
            continue
        try:
            PRESETS[cfg_file.stem] = yaml.safe_load(cfg_file.read_text())
        except Exception as exc:
            logger.warning("Failed to load preset %s: %s", cfg_file.name, exc)

    return PRESETS


# Eager load
load_presets()


def load_base_config(config_path: str | None = None) -> dict[str, Any]:
    """Load defaults.yaml or custom path."""
    if config_path is None:
        root_configs = Path(__file__).parent.parent / "configs"
        config_path = str(root_configs / "defaults.yaml")

        if not Path(config_path).exists():
            config_path = str(Path(__file__).parent / "configs" / "defaults.yaml")

    if not Path(config_path).exists():
        return {}

    with open(config_path) as f:
        result = yaml.safe_load(f)
        return result if isinstance(result, dict) else {}


def parse_override_args(args_list: list[str] | None) -> dict[str, Any]:
    """Parse key=value strings into typed dict."""
    if not args_list:
        return {}

    overrides: dict[str, Any] = {}
    for arg in args_list:
        if "=" not in arg:
            logger.warning("Skipping malformed override (no '='): %s", arg)
            continue

        key, raw = arg.split("=", 1)

        # Strip quotes
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]

        if raw.lower() in ("true", "false"):
            overrides[key] = raw.lower() == "true"
        elif raw.startswith("[") and raw.endswith("]"):
            try:
                overrides[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                logger.warning("Cannot parse list for '%s', keeping as string", key)
                overrides[key] = raw
        else:
            for cast in (int, float):
                try:
                    overrides[key] = cast(raw)
                    break
                except ValueError:
                    continue
            else:
                overrides[key] = raw

    return overrides


def merge_configs(
    defaults: dict[str, Any],
    preset: dict[str, Any],
    yaml_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge order: defaults < preset < yaml_config < overrides."""
    merged = defaults.copy()
    merged.update(preset)
    merged.update(yaml_config)
    merged.update(overrides)
    return merged


def config_dict_to_argv(
    config: dict[str, Any],
    valid_fields: set[str] | None = None,
) -> list[str]:
    """Convert config dict to CLI arguments list."""
    if valid_fields is None:
        valid_fields = _get_valid_fields()

    argv: list[str] = []
    for k, v in config.items():
        if k not in valid_fields or v is None:
            continue
        if isinstance(v, bool):
            if v: list[str] = []
    for k, v in config.items():
        if k not in valid_fields or v is None:
            continue
        if isinstance(v, bool):
            # HfArgumentParser only recognises --flag (store_true).
            # False booleans cannot be expressed as argv; they are
            # handled by apply_false_booleans() after parsing.
            if v:
                argv.append(f"--{k}")
        elif isinstance(v, list | tuple):
            argv.append(f"--{k}")
            argv.extend(str(item) for item in v)
        else:
            argv.append(f"--{k}")
            argv.append(str(v))
    return argv


def apply_false_booleans(
    config: dict[str, Any],
    dataclass_instances: tuple,
    cli_args: list[str],
) -> None:
    """Patch dataclass instances for False booleans that argv cannot express.

    Only applies when the CLI didn't explicitly set the field (i.e. the
    flag doesn't appear in *cli_args*).
    """
    for dc in dataclass_instances:
        for f in dataclasses.fields(dc):
            if (
                f.name in config
                and isinstance(config[f.name], bool)
                and not config[f.name]
                and f"--{f.name}" not in cli_args
            ):
                setattr(dc, f.name, False)
