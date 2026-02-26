# Configuration Module

Configuration loading and parsing utilities for the training pipeline.

**Location**: `vembed/training/config.py`

**Lines**: 60

## Overview

The config module handles:
- Loading configuration from YAML files
- Parsing CLI arguments
- Merging configurations (defaults → presets → files → CLI)
- Extracting distributed training settings

## Key Functions

### `load_and_parse_config()`

```python
def load_and_parse_config() -> dict:
    """
    Load and parse complete training configuration.

    Hierarchy (lowest to highest priority):
    1. Hardcoded defaults
    2. Preset YAML (from vembed/config.py)
    3. User-provided YAML file
    4. CLI overrides
    5. Special overrides (gradient_checkpointing)

    Returns:
        dict: Complete merged configuration
    """
```

### `get_distributed_config(config: dict) -> tuple`

```python
def get_distributed_config(config: dict) -> tuple[bool, bool, bool]:
    """
    Extract distributed training configuration.

    Returns:
        (use_grad_ckpt, use_grad_cache, find_unused_params)
    """
```

### `prepare_output_dir(config: dict) -> str`

```python
def prepare_output_dir(config: dict) -> str:
    """Prepare and return the output directory path."""
```

## Usage Example

```python
from vembed.training.config import load_and_parse_config, get_distributed_config

# Load all configuration sources
config = load_and_parse_config()

# Extract distributed training settings
use_grad_ckpt, use_grad_cache, find_unused = get_distributed_config(config)

print(f"Model: {config.model_name}")
print(f"Batch size: {config.batch_size}")
print(f"Gradient caching: {use_grad_cache}")
```

## Configuration Structure

See `vembed/config.py` for the complete list of configuration parameters. Common options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | N/A | HuggingFace model ID |
| `batch_size` | int | 32 | Training batch size per GPU |
| `learning_rate` | float | 2e-5 | AdamW learning rate |
| `epochs` | int | 3 | Number of training epochs |
| `use_gradient_cache` | bool | true | Enable memory optimization |
| `gradient_cache_chunk_size` | int | 32 | Chunk size for gradient cache |
| `use_lora` | bool | false | Enable LoRA |
| `use_mrl` | bool | false | Enable Matryoshka learning |

## Related Modules

- [data_utils.md](./data_utils.md) - Data handling
- [optimizer_builder.md](./optimizer_builder.md) - Optimizer creation
- [model_builder.md](./model_builder.md) - Model initialization
