"""Shared configuration utilities for vembed-factory.

Centralises config loading, merging, and serialisation so that both
``vembed.cli`` and ``vembed.entrypoints.train`` use the same logic.
"""

import ast
import dataclasses
import logging
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore

from vembed.hparams import DataArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps substrings in model names to their corresponding preset keys.
# Checked in order — first match wins.
_MODEL_NAME_PRESET_HINTS: list[tuple[str | tuple[str, ...], str]] = [
    ("clip", "clip"),
    ("siglip", "siglip"),
    (("qwen", "vl"), "qwen"),  # must contain both substrings
]

_VLM_KEYWORDS = ("vl", "vision", "intervl", "llava", "gemma")

_VLM_FALLBACK_PRESET: dict[str, Any] = {
    "encoder_mode": "vlm_generic",
    "batch_size": 1,
    "lr": 1.0e-5,
    "gradient_cache_chunk_size": 1,
}

# All valid field names across the three dataclasses.
_VALID_FIELDS: set[str] | None = None


def _get_valid_fields() -> set[str]:
    global _VALID_FIELDS
    if _VALID_FIELDS is None:
        _VALID_FIELDS = set()
        for dc in (ModelArguments, DataArguments, TrainingArguments):
            for f in dataclasses.fields(dc):
                _VALID_FIELDS.add(f.name)
    return _VALID_FIELDS


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

PRESETS: dict[str, Any] = {}


def load_presets() -> dict[str, Any]:
    """Discover and load all YAML preset files from the configs/ directory."""
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


# Eagerly load on import so PRESETS is populated.
load_presets()


# ---------------------------------------------------------------------------
# Base / defaults config
# ---------------------------------------------------------------------------


def load_base_config(config_path: str | None = None) -> dict[str, Any]:
    """Load ``defaults.yaml`` (or a custom path) and return as a dict."""
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


# ---------------------------------------------------------------------------
# Preset inference from model name
# ---------------------------------------------------------------------------


def infer_preset_from_model_name(model_name: str) -> dict[str, Any]:
    """Heuristically pick a preset by inspecting the model name."""
    lower = model_name.lower()

    for pattern, preset_key in _MODEL_NAME_PRESET_HINTS:
        if isinstance(pattern, tuple):
            matched = all(p in lower for p in pattern)
        else:
            matched = pattern in lower
        if matched:
            return cast(dict[str, Any], PRESETS.get(preset_key, {}))

    if any(kw in lower for kw in ("bert", "roberta", "bge")):
        return {}

    if any(kw in lower for kw in _VLM_KEYWORDS):
        return dict(_VLM_FALLBACK_PRESET)

    return {}


# ---------------------------------------------------------------------------
# Override parsing  (key=value strings → typed dict)
# ---------------------------------------------------------------------------


def parse_override_args(args_list: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` strings into a typed dictionary.

    Handles booleans, lists (via ``ast.literal_eval``), ints, floats,
    and falls back to plain strings.
    """
    if not args_list:
        return {}

    overrides: dict[str, Any] = {}
    for arg in args_list:
        if "=" not in arg:
            logger.warning("Skipping malformed override (no '='): %s", arg)
            continue

        key, raw = arg.split("=", 1)

        # Strip surrounding quotes
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


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------


def merge_configs(
    defaults: dict[str, Any],
    preset: dict[str, Any],
    yaml_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Single-pass merge: defaults < preset < yaml_config < overrides."""
    merged = defaults.copy()
    merged.update(preset)
    merged.update(yaml_config)
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# Dict → argv serialisation for HfArgumentParser
# ---------------------------------------------------------------------------


def config_dict_to_argv(
    config: dict[str, Any],
    valid_fields: set[str] | None = None,
) -> list[str]:
    """Convert a config dict to a list of CLI-style arguments.

    Only keys present in *valid_fields* (defaults to the union of all
    hparam dataclass fields) are emitted.  Booleans are serialised as
    ``--key`` / ``--no_key``, lists as ``--key v1 v2 …``, and everything
    else as ``--key value``.
    """
    if valid_fields is None:
        valid_fields = _get_valid_fields()

    argv: list[str] = []
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
