# Model Builder Module

Utilities for model initialization, LoRA setup, and optimization.

**Location**: `vembed/training/model_builder.py`

**Lines**: 200

## Overview

The model_builder module handles:
- Loading and initializing models from HuggingFace
- Applying LoRA (Parameter-Efficient Fine-Tuning)
- Enabling gradient checkpointing
- Torch.compile optimization
- Teacher model setup for knowledge distillation

## Key Functions

### `build_model(config) -> torch.nn.Module`

```python
def build_model(config) -> torch.nn.Module:
    """
    Build and initialize the student model.

    Features:
    - Loads from HuggingFace model hub
    - Applies LoRA if configured
    - Sets up gradient checkpointing
    - Configures torch_dtype

    Args:
        config: Configuration dict with:
            - model_name (str): HuggingFace model ID
            - use_lora (bool, default False)
            - lora_r (int, default 16)
            - lora_alpha (int, default 32)
            - use_gradient_checkpointing (bool, default False)
            - torch_dtype (str, optional: "float32", "float16", "bfloat16")

    Returns:
        torch.nn.Module: Initialized model
    """
```

### `build_teacher_model(config) -> torch.nn.Module | None`

```python
def build_teacher_model(config) -> torch.nn.Module | None:
    """
    Build teacher model for knowledge distillation.

    Returns None if teacher_model_name not in config.

    Args:
        config: Configuration dict with:
            - teacher_model_name (str, optional)
            - teacher_model_config (dict, optional)

    Returns:
        torch.nn.Module or None: Teacher model if configured
    """
```

### `apply_lora(model, config, accelerator) -> None`

```python
def apply_lora(
    model: torch.nn.Module,
    config: dict,
    accelerator,
) -> None:
    """
    Apply LoRA (Low-Rank Adaptation) to model.

    Reduces trainable parameters by:
    - Only training low-rank adapter layers
    - Freezing original weights
    - Reduces VRAM by ~10% typically

    Args:
        model: Model to apply LoRA to
        config: Configuration dict with:
            - use_lora (bool)
            - lora_r (int): LoRA rank
            - lora_alpha (int): LoRA scaling
            - lora_dropout (float, default 0.05)
            - lora_target_modules (list, optional)
        accelerator: Accelerate accelerator instance
    """
```

### `enable_static_graph(model, config, accelerator) -> None`

```python
def enable_static_graph(model, config, accelerator) -> None:
    """
    Enable DDP static graph optimization.

    For distributed training with DDP, allows PyTorch to optimize
    the computation graph when it's static (same structure each iteration).

    Args:
        model: Model to optimize
        config: Configuration with find_unused_parameters
        accelerator: Accelerate accelerator
    """
```

### `compile_model(model, config, accelerator) -> torch.nn.Module`

```python
def compile_model(
    model: torch.nn.Module,
    config: dict,
    accelerator,
) -> torch.nn.Module:
    """
    Optional torch.compile optimization (PyTorch 2.0+).

    Can improve performance 20-40% with PyTorch 2.0+ on compatible GPUs.

    Args:
        model: Model to compile
        config: Configuration with compile settings
        accelerator: Accelerate accelerator

    Returns:
        torch.nn.Module: Compiled or original model
    """
```

## Usage Examples

### Basic Model Loading

```python
from vembed.training.model_builder import build_model

config = {
    "model_name": "openai/clip-vit-base-patch32",
}

model = build_model(config)
```

### With LoRA

```python
from vembed.training.model_builder import build_model, apply_lora
from accelerate import Accelerator

config = {
    "model_name": "openai/clip-vit-large-patch14",
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
}

accelerator = Accelerator()
model = build_model(config)
apply_lora(model, config, accelerator)
```

### With Knowledge Distillation

```python
from vembed.training.model_builder import build_model, build_teacher_model

config = {
    "model_name": "openai/clip-vit-base-patch32",
    "teacher_model_name": "openai/clip-vit-large-patch14",
    "use_lora": False,
}

student_model = build_model(config)
teacher_model = build_teacher_model(config)

# Freeze teacher
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
```

### With All Optimizations

```python
from vembed.training.model_builder import (
    build_model,
    apply_lora,
    enable_static_graph,
    compile_model,
)
from accelerate import Accelerator

config = {
    "model_name": "qwen3-vl-embedding-2b",
    "use_lora": True,
    "lora_r": 16,
    "use_gradient_checkpointing": True,
    "compile": True,  # PyTorch 2.0+
}

accelerator = Accelerator()
model = build_model(config)

if config.get("use_lora"):
    apply_lora(model, config, accelerator)

enable_static_graph(model, config, accelerator)
model = compile_model(model, config, accelerator)
```

## Configuration Reference

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | N/A | HuggingFace model ID |
| `torch_dtype` | str | None | Precision: "float32", "float16", "bfloat16" |

### LoRA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora` | bool | false | Enable LoRA |
| `lora_r` | int | 16 | LoRA rank (lower = fewer params) |
| `lora_alpha` | int | 32 | LoRA scaling factor |
| `lora_dropout` | float | 0.05 | Dropout in LoRA layers |
| `lora_target_modules` | list | auto-detect | Which modules to apply LoRA to |

### Gradient Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gradient_checkpointing` | bool | false | Enable activation recomputation |
| `gradient_checkpointing_kwargs` | dict | {} | Additional checkpointing options |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compile` | bool | false | Use torch.compile (PyTorch 2.0+) |
| `compile_mode` | str | "default" | Compile mode: "default", "reduce-overhead", "max-autotune" |

## Performance Impact

### LoRA
- **VRAM**: ~10% reduction typically
- **Speed**: Slightly faster (fewer parameters to train)
- **Quality**: Minor - usually <1% reduction in performance

### Gradient Checkpointing
- **VRAM**: ~20% reduction
- **Speed**: ~20% slower (recomputation cost)
- **Quality**: No impact (mathematically equivalent)

### Torch Compile
- **VRAM**: <1% change
- **Speed**: +20-40% faster (PyTorch 2.0+, compatible GPUs)
- **Quality**: No impact (algorithmic improvement)

## Related Modules

- [config.md](./config.md) - Configuration loading
- [optimizer_builder.md](./optimizer_builder.md) - Optimizer creation
- [training_loop.md](./training_loop.md) - Used in Trainer initialization
- [checkpoint.md](./checkpoint.md) - Model saving
