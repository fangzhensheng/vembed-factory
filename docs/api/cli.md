# CLI Interface

Command-line interface for training and dataset utilities.

## Overview

The CLI handles YAML loading, CLI overrides, dataset helpers, and training launch orchestration.

### Key Functions

| Function | Purpose |
|----------|---------|
| `main()` | Entry point for the CLI |
| `print_usage()` | Print top-level help |

## Quick Start

### Training a Model

```bash
# Using a YAML config file
vembed train examples/quickstart/clip_minimal.yaml

# Override specific parameters
vembed train examples/quickstart/clip_minimal.yaml --batch_size 64 --learning_rate 1e-5

# Distributed configuration example
vembed train examples/distributed/qwen3_vl_8b_fsdp.yaml
```

### Python Usage

```python
from vembed.cli import main

args = [
    "train",
    "examples/quickstart/clip_minimal.yaml",
    "--batch_size", "64",
    "--learning_rate", "1e-5",
]
main(args)
```

## Common Workflows

### Quick Start with Defaults

```bash
vembed train examples/quickstart/clip_minimal.yaml   --data_path your_data.jsonl   --output_dir ./my_model
```

### Fine-tune a Vision-Language Model

```bash
vembed train examples/quickstart/qwen3_vl_minimal.yaml   --data_path data/train.jsonl   --epochs 10   --use_gradient_cache
```

### Training Config Priority

The CLI respects this priority order:

1. CLI arguments
2. User YAML config
3. Base defaults

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| positional `config.yaml` | Path to the YAML config file |
| `--batch_size` | Batch size per device |
| `--epochs` | Number of training epochs |
| `--learning_rate` | AdamW learning rate |
| `--output_dir` | Directory to save checkpoints |
| `--report_to` | Tracker: `wandb`, `swanlab`, `tensorboard`, `none` |
| `--dry_run` | Generate the merged config without launching training |

## Related Modules

- [High-Level Trainer](training/trainer.md)
- [Training Entrypoints](entrypoints.md)
- [Configuration](config.md)

## API Reference

::: vembed.cli.main
