# Optimizer Builder Module

Utilities for creating optimizers, schedulers, and experiment trackers.

**Location**: `vembed/training/optimizer_builder.py`

**Lines**: 110

## Overview

The optimizer_builder module handles:
- AdamW optimizer creation with selective weight decay
- Learning rate scheduler setup with warmup
- Experiment tracker configuration (W&B, TensorBoard, SwanLab)
- Parameter group management

## Key Functions

### `build_optimizer(model, config) -> torch.optim.Optimizer`

```python
def build_optimizer(model, config) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with selective weight decay.

    Applies weight decay to:
    - weight matrices (LayerNorm, embedding weights excluded)

    Skips weight decay for:
    - biases
    - LayerNorm parameters
    - embedding weights

    Args:
        model: PyTorch model to optimize
        config: Configuration dict with:
            - learning_rate (float, default 2e-5)
            - weight_decay (float, default 0.01)
            - adam_epsilon (float, default 1e-8)

    Returns:
        torch.optim.AdamW: Configured optimizer
    """
```

### `build_scheduler(optimizer, config, num_epochs, steps_per_epoch) -> tuple`

```python
def build_scheduler(
    optimizer,
    config,
    num_epochs: int,
    steps_per_epoch: int,
) -> tuple:
    """
    Build learning rate scheduler with warmup.

    Supports:
    - cosine: Cosine annealing with warmup (default)
    - linear: Linear warmup → linear decay
    - constant: Constant LR with warmup
    - constant_with_warmup: Warmup then constant

    Args:
        optimizer: AdamW optimizer
        config: Configuration dict with:
            - scheduler_type (str, default "cosine")
            - warmup_ratio (float, default 0.1)
            - learning_rate (float)
        num_epochs: Total training epochs
        steps_per_epoch: Steps per epoch

    Returns:
        (scheduler, warmup_steps): LR scheduler and warmup step count
    """
```

### `resolve_tracker(report_to) -> tuple`

```python
def resolve_tracker(report_to: str) -> tuple[str, dict]:
    """
    Resolve experiment tracker configuration.

    Supported trackers:
    - wandb: Weights & Biases
    - tensorboard: PyTorch TensorBoard
    - swanlab: SwanLab experiment tracker
    - none: No tracking

    Args:
        report_to: Tracker name

    Returns:
        (log_with, init_kwargs): Tracker name and initialization args
    """
```

## Usage Example

```python
from vembed.training.optimizer_builder import (
    build_optimizer,
    build_scheduler,
    resolve_tracker,
)

# Create optimizer
optimizer = build_optimizer(model, config)

# Create scheduler
scheduler, warmup_steps = build_scheduler(
    optimizer,
    config,
    num_epochs=10,
    steps_per_epoch=1000,
)

print(f"Total steps: {10 * 1000}")
print(f"Warmup steps: {warmup_steps}")

# Setup experiment tracking
log_with, init_kwargs = resolve_tracker("wandb")
```

## Configuration Reference

### Optimizer Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 2e-5 | Initial learning rate |
| `weight_decay` | float | 0.01 | Weight decay (L2 regularization) |
| `adam_epsilon` | float | 1e-8 | Epsilon for Adam stability |

### Scheduler Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scheduler_type` | str | cosine | LR schedule type |
| `warmup_ratio` | float | 0.1 | Fraction of training for warmup |

### Scheduler Types

- **cosine**: Cosine annealing with linear warmup
  - Good for general-purpose training
  - Recommended default

- **linear**: Linear decay with warmup
  - Monotonic LR decrease
  - Suitable for fine-tuning

- **constant**: Constant LR with warmup
  - Training at fixed LR after warmup
  - For stable fine-tuning

- **constant_with_warmup**: Synonym for constant

## Experiment Tracking

### Weights & Biases (W&B)

```bash
wandb login
python train.py config.yaml --config_override report_to=wandb
```

Logs to W&B with:
- Training/validation metrics
- Learning rate schedule
- Model architecture
- Configuration parameters

### TensorBoard

```bash
python train.py config.yaml --config_override report_to=tensorboard
tensorboard --logdir output/runs
```

### SwanLab

```bash
python train.py config.yaml --config_override report_to=swanlab
```

## Related Modules

- [model_builder.md](./model_builder.md) - Model setup (parameters for optimizer)
- [training_loop.md](./training_loop.md) - Used in Trainer initialization
