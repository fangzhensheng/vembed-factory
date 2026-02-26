# Checkpoint Module

Utilities for saving and managing model checkpoints.

**Location**: `vembed/training/checkpoint.py`

**Lines**: 60

## Overview

The checkpoint module handles:
- Saving model checkpoints during training
- Persisting vembed-specific configuration
- Saving processor/tokenizer for inference
- Checkpoint organization

## Key Functions

### `save_checkpoint(path, model, accelerator, processor, config) -> None`

```python
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    accelerator,
    processor,
    config: dict,
    prefix: str = "",
) -> None:
    """
    Save model checkpoint with vembed configuration.

    Saves:
    1. Model weights (via accelerator)
    2. Processor/tokenizer
    3. vembed configuration (as JSON)

    Args:
        path (str): Directory to save checkpoint
        model (torch.nn.Module): Model to save
        accelerator: Accelerate accelerator
        processor: Processor/tokenizer for inference
        config (dict): vembed configuration
        prefix (str): Optional prefix for checkpoint name
    """
```

## Usage Example

```python
from vembed.training.checkpoint import save_checkpoint

# Save checkpoint during training
save_checkpoint(
    path=f"output/checkpoint-step-{global_step}",
    model=model,
    accelerator=accelerator,
    processor=processor,
    config=config,
)

# Save final model
save_checkpoint(
    path="output/final",
    model=model,
    accelerator=accelerator,
    processor=processor,
    config=config,
    prefix="model",
)
```

## Checkpoint Structure

After `save_checkpoint()`, the directory contains:

```
output/checkpoint-1000/
├── model_state_dict.safetensors   # Model weights
├── processor_config.json            # Processor configuration
├── preprocessor_config.json         # Image preprocessor config
├── tokenizer.json                   # Tokenizer (if text model)
├── special_tokens_map.json          # Special tokens
├── vembed_config.json               # vembed-specific settings
└── config.json                      # HF model config (from accelerator)
```

## vembed Configuration

The saved `vembed_config.json` contains:
- Model name
- Training configuration
- Data preprocessing settings
- Retrieval mode
- Distillation settings (if applicable)

This allows full reproducibility of the trained model.

## Related Modules

- [config.md](./config.md) - Configuration to save
- [training_loop.md](./training_loop.md) - Called during training
- [model_builder.md](./model_builder.md) - Model structure
