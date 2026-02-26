# Training Module API Documentation

Complete API reference for the refactored `vembed.training` module with 8 specialized components.

## Module Overview

The training module has been reorganized into focused, testable components:

| Module | Purpose | Lines | See |
|--------|---------|-------|-----|
| **config.py** | Configuration loading & parsing | 60 | [config.md](./config.md) |
| **data_utils.py** | Batch unpacking & concatenation | 220 | [data_utils.md](./data_utils.md) |
| **optimizer_builder.py** | Optimizer & scheduler creation | 110 | [optimizer_builder.md](./optimizer_builder.md) |
| **model_builder.py** | Model initialization with optimizations | 200 | [model_builder.md](./model_builder.md) |
| **checkpoint.py** | Checkpoint saving & management | 60 | [checkpoint.md](./checkpoint.md) |
| **evaluator.py** | Validation & evaluation loop | 130 | [evaluator.md](./evaluator.md) |
| **training_loop.py** | Core Trainer class | 490 | [training_loop.md](./training_loop.md) |

**Total**: 1,265 lines across 8 focused modules

---

## Quick Start

### Minimal Example

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer
from accelerate import Accelerator

# Load configuration
config = load_and_parse_config()

# Build components
accelerator = Accelerator()
model = build_model(config)
optimizer = build_optimizer(model, config)

# Create and run trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
)
trainer.train()
```

### With All Features

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model, build_teacher_model, apply_lora
from vembed.training.optimizer_builder import build_optimizer, build_scheduler
from vembed.training.evaluator import Evaluator
from accelerate import Accelerator

# Setup
config = load_and_parse_config()
accelerator = Accelerator()

# Build student model with LoRA
student_model = build_model(config)
if config.get("use_lora"):
    apply_lora(student_model, config, accelerator)

# Build teacher model (if distillation)
teacher_model = None
if config.get("teacher_model_name"):
    teacher_model = build_teacher_model(config)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

# Build optimizer and scheduler
optimizer = build_optimizer(student_model, config)
num_epochs = config.get("epochs", 3)
steps_per_epoch = len(train_loader)
scheduler, _ = build_scheduler(optimizer, config, num_epochs, steps_per_epoch)

# Create evaluator (if validation data exists)
evaluator = None
if val_loader:
    evaluator = Evaluator(student_model, criterion, accelerator, config)

# Create trainer
trainer = Trainer(
    model=student_model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
    evaluator=evaluator,
    teacher_model=teacher_model,
)

# Train
trainer.train()
```

---

## Module Dependency Graph

```
load_and_parse_config() [config.py]
    ↓
build_model() [model_builder.py]
    ↓
apply_lora() [model_builder.py]
    ↓
build_optimizer() [optimizer_builder.py]
    ↓
build_scheduler() [optimizer_builder.py]
    ↓
Trainer() [training_loop.py]
    ├── uses data_utils.py (unpack_query_batch, concat_batches)
    ├── uses checkpoint.py (save_checkpoint)
    ├── uses evaluator.py (Evaluator.evaluate)
    └── uses config for all settings
```

---

## Common Use Cases

### 1. Simple CLI Training

```bash
accelerate launch vembed/entrypoints/train.py --config config.yaml
```

No code changes needed! Uses all modules internally.

### 2. Programmatic Training

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model

config = load_and_parse_config()
model = build_model(config)
trainer = Trainer(model=model, ...)
trainer.train()
```

### 3. Custom Training Loop

```python
from vembed.training import Trainer

class MyTrainer(Trainer):
    def _step_standard(self, batch):
        # Your custom training step
        loss = super()._step_standard(batch)
        return loss

trainer = MyTrainer(...)
trainer.train()
```

### 4. Advanced Component Usage

```python
# Use individual components independently
from vembed.training.config import load_and_parse_config
from vembed.training.data_utils import unpack_query_batch, concat_batches
from vembed.training.model_builder import build_model, apply_lora

config = load_and_parse_config()
model = build_model(config)
apply_lora(model, config, accelerator)

# Manually process data
query_batch = unpack_query_batch(batch, config['retrieval_mode'])
model_output = model(query_batch)
```

---

## Migration Guide

### From monolithic train.py (790 lines)

**Before** (all in one file):
```python
# train.py: 790 lines of mixed concerns
def parse_args(): ...
def unpack_query_batch(): ...
def build_model(): ...
def main(): ... # 500+ lines of orchestration
```

**After** (refactored into modules):
```python
# Use specific modules as needed
from vembed.training.config import load_and_parse_config
from vembed.training.data_utils import unpack_query_batch
from vembed.training.model_builder import build_model
from vembed.training import Trainer
```

✅ **100% backward compatible** - CLI commands work unchanged
✅ **Better testability** - Each module can be tested independently
✅ **Better reusability** - Import specific components
✅ **Clearer code** - Each file has single responsibility

---

## Performance & Optimization

### Memory Optimization

Combine multiple techniques:
1. **Gradient Cache** (config.py `use_gradient_cache=true`)
   - 40% VRAM reduction typical
   - Effective batch size scaling

2. **Gradient Checkpointing** (model_builder.py via config)
   - 20% VRAM reduction
   - Trades compute for memory

3. **LoRA** (model_builder.py via config)
   - 10% parameter reduction
   - Faster training

### Speed Optimization

1. **Torch Compile** (PyTorch 2.0+)
   - 20-40% speedup on compatible GPUs

2. **Mixed Precision** (via accelerator)
   - FP16/BF16 training

3. **Distributed Training** (via accelerator DDP)
   - Multi-GPU scaling

---

## Troubleshooting

### Import Errors

```python
# ✓ Correct
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model

# ✗ Wrong - VEmbedFactoryTrainer is from vembed.trainer, not training
from vembed.training import VEmbedFactoryTrainer  # ImportError!
# Use instead:
from vembed.trainer import VEmbedFactoryTrainer
```

### Configuration Issues

```python
# Load all configuration sources
config = load_and_parse_config()

# Check what's in config
print(config.keys())

# Verify specific settings
print(f"Model: {config.get('model_name')}")
print(f"Batch size: {config.get('batch_size')}")
```

### GPU Memory Issues

```python
# Enable gradient caching
config['use_gradient_cache'] = True
config['gradient_cache_chunk_size'] = 1

# Enable gradient checkpointing
config['use_gradient_checkpointing'] = True

# Apply LoRA
config['use_lora'] = True
config['lora_r'] = 8  # Smaller rank
```

---

## Related Documentation

- [REFACTORING_SUMMARY.md](../../REFACTORING_SUMMARY.md) - Complete refactoring overview
- [QUICK_REFERENCE.md](../../QUICK_REFERENCE.md) - Quick module reference
- [vembed/training/README.md](../../vembed/training/README.md) - User guide with examples
- [trainer.md](./trainer.md) - VEmbedFactoryTrainer (high-level API)

---

## API Reference Index

### Configuration
- [config.load_and_parse_config()](./config.md#load_and_parse_config)
- [config.get_distributed_config()](./config.md#get_distributed_config)

### Data Processing
- [data_utils.unpack_query_batch()](./data_utils.md#unpack_query_batch)
- [data_utils.concat_batches()](./data_utils.md#concat_batches)
- [data_utils.maybe_first()](./data_utils.md#maybe_first)

### Model Building
- [model_builder.build_model()](./model_builder.md#build_model)
- [model_builder.apply_lora()](./model_builder.md#apply_lora)
- [model_builder.compile_model()](./model_builder.md#compile_model)

### Optimization
- [optimizer_builder.build_optimizer()](./optimizer_builder.md#build_optimizer)
- [optimizer_builder.build_scheduler()](./optimizer_builder.md#build_scheduler)

### Training
- [training_loop.Trainer](./training_loop.md#trainer)
- [training_loop.Trainer.train()](./training_loop.md#train)

### Evaluation
- [evaluator.Evaluator](./evaluator.md#evaluator)
- [evaluator.Evaluator.evaluate()](./evaluator.md#evaluate)

### Checkpointing
- [checkpoint.save_checkpoint()](./checkpoint.md#save_checkpoint)

---

**Last Updated**: 2026-02-26
**Version**: 1.0
**Compatibility**: 100% backward compatible with original train.py
