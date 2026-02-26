# Hyperparameters

Configuration dataclasses for vembed-factory.

## Overview

The hyperparameters module defines all configuration dataclasses using Python's `dataclasses` module. These provide type hints, defaults, and validation for all training and inference parameters.

### Key Classes

| Class | Purpose |
|-------|---------|
| `ModelArguments` | Model loading parameters |
| `DataArguments` | Data loading parameters |
| `TrainingArguments` | Training hyperparameters |

## Quick Start

```python
from vembed.hparams import TrainingArguments

config = TrainingArguments(
    model_name="openai/clip-vit-base-patch32",
    batch_size=128,
    num_train_epochs=3,
    learning_rate=5e-5,
    output_dir="output"
)

print(f"Batch size: {config.batch_size}")
```

## Common Patterns

### From YAML to Python
```python
import yaml
from vembed.hparams import TrainingArguments

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

args = TrainingArguments(**config_dict)
```

### Modifying Arguments
```python
args = TrainingArguments(...)

# Override specific arguments
args.batch_size = 256
args.learning_rate = 1e-5
```

## Configuration Hierarchy

```
YAML Config
    ↓
DataClass Defaults
    ↓
CLI Arguments (override everything)
```

---

::: vembed.hparams.ModelArguments
::: vembed.hparams.DataArguments
::: vembed.hparams.TrainingArguments
