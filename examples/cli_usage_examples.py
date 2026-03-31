"""Examples demonstrating the new vembed-factory CLI usage.

This file shows how to use both the modern Typer-based CLI and
the legacy CLI for backward compatibility.

Installation:
    pip install -e .

Then you can use the CLI:
    vembed train --help
    vembed validate-data --help
    vembed show-config
"""

# ============================================================================
# MODERN TYPER CLI EXAMPLES
# ============================================================================

"""
1. Train with a config file:
   vembed train config.yaml

2. Train with CLI overrides:
   vembed train config.yaml --batch-size 64 --lr 1e-5 --epochs 5

3. Train with quick-start template (no config file needed):
   vembed train --quick-start clip --data-path data/train.jsonl

4. Quick-start with overrides:
   vembed train --quick-start siglip --data-path data/train.jsonl --num-gpus 2

5. Resume from checkpoint:
   vembed train config.yaml --resume-from output/checkpoint-100

6. Dry run (generate config without launching training):
   vembed train config.yaml --dry-run

7. Validate dataset before training:
   vembed validate-data data/train.jsonl --check-images --image-root images/

8. Show available templates:
   vembed list-templates

9. Display configuration (defaults + file):
   vembed show-config config.yaml

10. Enable FSDP for large model training:
    vembed train config.yaml --use-fsdp --num-gpus 8

11. Enable LoRA for parameter-efficient fine-tuning:
    vembed train config.yaml --use-lora

12. Verbose logging:
    vembed train config.yaml --verbose
"""

# ============================================================================
# LEGACY CLI EXAMPLES (Still Supported)
# ============================================================================

"""
These command formats are still supported for backward compatibility:

1. Basic training with config:
   vembed config.yaml

2. Training with CLI overrides:
   vembed config.yaml --learning_rate=1e-5 --batch_size=64

3. Validate data (legacy format):
   vembed validate-data data/train.jsonl

Note: The legacy format uses underscores (_) while modern Typer uses hyphens (-)
"""

# ============================================================================
# EXAMPLE CONFIG FILE (config.yaml)
# ============================================================================

EXAMPLE_CONFIG_YAML = """
# Model Configuration
model_name_or_path: openai/clip-vit-base-patch32
encoder_mode: clip
pooling_method: cls

# Data Configuration
data_path: data/train.jsonl
val_data_path: data/val.jsonl
image_root: images/
column_mapping:
  query: text
  positive: image
  negatives: neg_images

# Training Configuration
output_dir: output/clip-baseline
batch_size: 32
learning_rate: 2.0e-5
epochs: 3

# Loss & Retrieval
loss_type: infonce
temperature: 0.05
retrieval_mode: t2i

# Optimization
optimizer: adamw
scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0

# Experiment Tracking
report_to: wandb
run_name: clip-baseline-v1
run_tags: [baseline, clip]

# Advanced Options (optional)
gradient_checkpointing: false
use_gradient_cache: true
gradient_cache_chunk_size: 4
"""

# ============================================================================
# EXAMPLE USAGE IN PYTHON
# ============================================================================

def example_parse_args():
    """Example: Parse arguments programmatically."""
    from vembed.hparams import parse_args, validate_args, ValidationError

    # Parse from config file and CLI overrides
    config = parse_args(
        config_path="config.yaml",
        cli_args=["--batch-size=64", "--learning_rate=1e-5"],
        defaults={"epochs": 3},
    )

    try:
        validate_args(config)
        print("Config validation passed!")
    except ValidationError as e:
        print(f"Config validation failed: {e}")

    return config


def example_cli_programmatic():
    """Example: Call CLI programmatically."""
    from vembed.cli import main

    # Train with config file
    exit_code = main(["train", "config.yaml"])

    # Train with quick-start
    exit_code = main(["train", "--quick-start", "clip", "--data-path", "data.jsonl"])

    # Validate data
    exit_code = main(["validate-data", "data.jsonl", "--check-images"])

    return exit_code


def example_config_operations():
    """Example: Config file operations."""
    from vembed.hparams import (
        load_config_file,
        parse_cli_overrides,
        merge_configs,
        flatten_nested_config,
        unflatten_config,
    )

    # Load config
    config = load_config_file("config.yaml")

    # Parse CLI args
    overrides = parse_cli_overrides(["--batch-size=64", "--use-lora"])

    # Merge configs
    merged = merge_configs({"epochs": 3}, config, overrides)

    # Flatten nested config (e.g., for logging)
    flat = flatten_nested_config({"model": {"name": "clip", "dim": 512}})
    # flat = {"model__name": "clip", "model__dim": 512}

    # Unflatten back
    nested = unflatten_config(flat)
    # nested = {"model": {"name": "clip", "dim": 512}}

    return merged


def example_validation():
    """Example: Individual validation checks."""
    from vembed.hparams import (
        validate_batch_size,
        validate_learning_rate,
        validate_lora_params,
        validate_encoder_mode,
        ValidationError,
    )

    try:
        validate_batch_size(32)
        validate_learning_rate(2e-5)
        validate_lora_params(lora_r=16, lora_alpha=32)
        validate_encoder_mode("clip")
        print("All validations passed!")
    except ValidationError as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    print(__doc__)
