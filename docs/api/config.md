# Configuration Management

Unified configuration system for training and inference.

## Overview

The configuration module manages all hyperparameters and settings for vembed-factory. It provides a consistent interface for loading, validating, and applying configurations from YAML files or Python dictionaries.

### Key Features
- YAML-based configuration
- Type validation and defaults
- Dataclass-based configuration objects
- Easy programmatic configuration

## Quick Start

### Load from YAML
```python
from vembed.config import load_config

# Load from YAML file
config = load_config("examples/clip_train.yaml")

# Access parameters
print(config.batch_size)
print(config.learning_rate)
```

### Create Programmatically
```python
from vembed.hparams import TrainingConfig

config = TrainingConfig(
    model_name="openai/clip-vit-base-patch32",
    batch_size=128,
    epochs=3,
    learning_rate=5e-5,
    output_dir="output"
)
```

## Configuration Parameters

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Batch size per GPU |
| `epochs` | int | 3 | Number of training epochs |
| `learning_rate` | float | 2e-5 | Adam learning rate |
| `warmup_ratio` | float | 0.1 | Fraction of steps for warmup |
| `scheduler_type` | str | `cosine` | LR scheduler: `cosine`, `linear`, `constant`, `constant_with_warmup` |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `max_grad_norm` | float | 1.0 | Gradient clipping norm |

### Memory Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gradient_cache` | bool | true | Enable gradient caching for large batches |
| `gradient_cache_chunk_size` | int | 32 | Chunk size for gradient cache |
| `gradient_checkpointing` | bool | false | Enable activation recomputation |
| `use_lora` | bool | false | Enable LoRA fine-tuning |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | - | HuggingFace model ID or local path |
| `model_type` | str | `auto` | Model type: `auto`, `custom`, `clip`, `qwen` |
| `attn_implementation` | str | `None` | Attention: `flash_attention_2`, `sdpa`, `eager` |
| `torch_dtype` | str | `bfloat16` | Precision: `float32`, `float16`, `bfloat16` |

## Configuration Priority

Configurations are loaded in this order (lower priority → higher priority):

```
defaults.yaml
    ↓
preset config (e.g., configs/clip.yaml)
    ↓
user YAML file
    ↓
CLI arguments (--batch_size 64)
```

## Common Examples

### YAML Configuration
```yaml
# Model
model_name: "openai/clip-vit-base-patch32"
model_type: "auto"

# Training
batch_size: 128
epochs: 3
learning_rate: 5.0e-5

# Memory Optimization
use_gradient_cache: true
gradient_cache_chunk_size: 64
use_lora: true
```

### Loading & Modifying
```python
from vembed.config import load_config

config = load_config("config.yaml")

# Override specific parameters
config.batch_size = 64
config.learning_rate = 1e-5

# Use in training
trainer = Trainer(config)
```

## FAQs

**Q: What's the recommended batch size?**
A: 128-256 for dual-encoders, 32-64 for large VLMs. Use gradient_cache to enable larger batches on smaller GPUs.

**Q: Should I use bfloat16 or float32?**
A: Use `bfloat16` with flash_attention_2 for best performance. Use `float32` only if you encounter numerical stability issues.

**Q: What learning rate should I use?**
A: Start with 5e-5 for fine-tuning. Use smaller LR (2e-5) for large models to avoid divergence.

## Related Modules

- **Hyperparameters**: [hparams.md](hparams.md) - Configuration dataclasses
- **CLI**: [cli.md](cli.md) - Command-line interface
- **Training**: [../training/trainer.md](../training/trainer.md) - Training orchestration

---

::: vembed.config
