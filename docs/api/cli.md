# CLI Interface

Command-line interface for training and evaluating embedding models.

## Overview

The CLI module provides the main entry point for vembed-factory. It handles argument parsing, configuration loading, and delegates to appropriate training/evaluation commands.

### Key Features
- Simple one-command training
- YAML configuration support
- CLI argument overrides
- Multi-GPU distributed training support

### Key Functions

| Function | Purpose |
|----------|---------|
| `main()` | Entry point for the CLI |
| `parse_args()` | Parse command-line arguments |

## Quick Start

### Training a Model

```bash
# Using a YAML config file
python run.py examples/clip_train.yaml

# Override specific parameters via CLI
python run.py examples/clip_train.yaml --batch_size 64 --learning_rate 1e-5

# Multi-GPU training
accelerate launch --multi_gpu --num_processes 4 run.py examples/clip_train.yaml
```

### Python API

```python
from vembed import Trainer

# Create trainer
trainer = Trainer("openai/clip-vit-base-patch32")

# Train
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    epochs=3,
    batch_size=128
)
```

## Common Workflows

### Workflow 1: Quick Start with Defaults
```bash
# Train CLIP on your data with default settings
python run.py examples/clip_train.yaml \
  --data_path your_data.jsonl \
  --output_dir ./my_model
```

### Workflow 2: Fine-tune Vision-Language Model
```bash
# Train Qwen3-VL with memory optimization
python run.py examples/qwen3_2b_train.yaml \
  --data_path data/train.jsonl \
  --epochs 10
```

### Workflow 3: Distributed Training
```bash
# Train on 8 GPUs with gradient cache
accelerate launch --multi_gpu --num_processes 8 run.py examples/clip_train.yaml
```

## Configuration Hierarchy

The CLI respects this priority order (highest to lowest):
1. **CLI Arguments** (e.g., `--batch_size 64`)
2. **User YAML Config** (e.g., `examples/my_config.yaml`)
3. **Preset YAML** (e.g., `configs/clip.yaml`)
4. **Default Config** (`configs/defaults.yaml`)

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | N/A | Path to YAML config file |
| `--batch_size` | 32 | Batch size per GPU |
| `--epochs` | 3 | Number of training epochs |
| `--learning_rate` | 2e-5 | Adam learning rate |
| `--output_dir` | `output` | Directory to save checkpoints |
| `--report_to` | `none` | Experiment tracker: `wandb`, `tensorboard`, `none` |

## Troubleshooting

**Q: "No such file or directory: examples/clip_train.yaml"**
A: Make sure you're running from the project root directory.

**Q: CUDA out of memory?**
A: Try enabling `use_gradient_cache: true` and reducing `batch_size` in your config.

**Q: How to resume training from checkpoint?**
A: Set `output_dir` to a new folder and ensure the old checkpoint has `training_args.bin`.

## Related Modules

- **Trainer API**: [trainer.md](../training/trainer.md) - High-level Python training interface
- **Configuration**: [config.md](config.md) - Configuration management
- **Training**: [entrypoints.md](entrypoints.md) - Training entry points

---

::: vembed.cli.main
::: vembed.cli.parse_args
