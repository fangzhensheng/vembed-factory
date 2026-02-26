# Training Module

Modularized training components for vembed-factory.

## Overview

The training module breaks down the original monolithic `train.py` into specialized, reusable components:

- **config.py** - Configuration management
- **data_utils.py** - Data processing utilities
- **optimizer_builder.py** - Optimizer and scheduler builders
- **model_builder.py** - Model initialization and setup
- **checkpoint.py** - Checkpoint management
- **evaluator.py** - Evaluation and validation
- **training_loop.py** - Core training loop

> **Note**: This `Trainer` class is different from `vembed.trainer.VEmbedFactoryTrainer` (the high-level Python API). This is the low-level training loop executor.

## Quick Start

### Using the CLI (Unchanged)

```bash
accelerate launch vembed/entrypoints/train.py \
    --config configs/train.yaml \
    --config_override model_name=openai/clip-vit-base-patch32 \
    --gradient_checkpointing
```

### Using the Python API

```python
from vembed.training import load_and_parse_config, Trainer
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer

# Load configuration
config = load_and_parse_config()

# Build components
model = build_model(config)
optimizer = build_optimizer(model, config)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
)

# Train
trainer.train()
```

## Module Details

### config.py

Configuration loading and management.

```python
from vembed.training.config import load_and_parse_config, get_distributed_config

# Load all configurations (file + CLI overrides)
config = load_and_parse_config()

# Get distributed training config
use_grad_ckpt, use_grad_cache, find_unused = get_distributed_config(config)
```

### data_utils.py

Data batch processing utilities.

```python
from vembed.training.data_utils import (
    unpack_query_batch,
    unpack_positive_batch,
    concat_batches,
    maybe_first,
)

# Extract query inputs from batch
q_inputs = unpack_query_batch(batch, retrieval_mode="t2i")

# Extract positive inputs
p_inputs = unpack_positive_batch(batch, retrieval_mode="t2i")

# Concatenate batches (for unified models like Qwen-VL)
concat_input, batch_sizes = concat_batches([q_inputs, p_inputs])

# Extract tensor from model output
embs = maybe_first(model_output)
```

### optimizer_builder.py

Optimizer and scheduler creation.

```python
from vembed.training.optimizer_builder import (
    build_optimizer,
    build_scheduler,
    resolve_tracker,
)

# Create optimizer with weight decay for non-bias parameters
optimizer = build_optimizer(model, config)

# Create scheduler with warmup
scheduler, warmup_steps = build_scheduler(
    optimizer, config, num_epochs=10, steps_per_epoch=100
)

# Resolve experiment tracker
log_with, init_kwargs = resolve_tracker(report_to="wandb")
```

### model_builder.py

Model initialization and setup.

```python
from vembed.training.model_builder import (
    build_model,
    build_teacher_model,
    apply_lora,
    enable_static_graph,
)

# Build student model
model = build_model(config)

# Build teacher model for distillation
teacher = build_teacher_model(config)

# Apply LoRA
apply_lora(model, config, accelerator)

# Enable static graph optimization for DDP
enable_static_graph(model, config, accelerator)
```

### checkpoint.py

Checkpoint saving and management.

```python
from vembed.training.checkpoint import save_checkpoint

# Save checkpoint with model, processor, and vembed config
save_checkpoint(
    path="./checkpoints/step-1000",
    model=model,
    accelerator=accelerator,
    processor=processor,
    config=config,
)
```

### evaluator.py

Validation and evaluation.

```python
from vembed.training.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(
    model=model,
    criterion=criterion,
    accelerator=accelerator,
    retrieval_mode="t2i",
    log_with="wandb",
)

# Run validation
avg_loss = evaluator.evaluate(val_dataloader, global_step=100)
```

### trainer.py

Core training loop.

```python
from vembed.training.trainer import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
    teacher_model=teacher_model,
    distillation_loss_fn=distill_fn,
    evaluator=evaluator,
    val_dataloader=val_loader,
)

# Run training
trainer.train()
```

## Features

### ✅ Gradient Caching

Automatic memory-efficient training:

```python
config = {
    "use_gradient_cache": True,
    "gradient_cache_chunk_size": 1,
}
```

### ✅ Knowledge Distillation

Train student model using teacher outputs:

```python
config = {
    "teacher_model_name": "openai/clip-vit-large-patch14",
    "distillation_alpha": 0.5,
}
```

### ✅ LoRA Fine-tuning

Parameter-efficient tuning:

```python
config = {
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
}
```

### ✅ Distributed Training

Seamless DDP support:

```bash
accelerate launch --multi_gpu vembed/entrypoints/train.py --config config.yaml
```

### ✅ Experiment Tracking

Multiple trackers supported:

```python
config = {
    "report_to": "wandb",  # or tensorboard, swanlab
}
```

## Architecture

```
┌─────────────────────────────────────────┐
│     train.py (Entrypoint)               │
│     - Orchestration                     │
│     - Component coordination            │
└────────┬────────────────────────────────┘
         │
    ┌────┴───────────────────────────────┐
    │                                    │
┌───▼──────┐  ┌────────────────┐  ┌─────▼────┐
│ config   │  │ model_builder  │  │ optimizer│
│ Loading  │  │ & setup        │  │ builder  │
└───┬──────┘  └────────┬───────┘  └─────┬────┘
    │                  │                │
    └──────────────────┼────────────────┘
                       │
            ┌──────────▼──────────┐
            │   Trainer           │
            │ (Training Loop)     │
            └──────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐  ┌─────▼─────┐  ┌────▼────┐
    │ data   │  │ checkpoint│  │evaluator│
    │ utils  │  │           │  │         │
    └────────┘  └───────────┘  └─────────┘
```

## Advanced Usage

### Custom Training Step

```python
class CustomTrainer(Trainer):
    def _step_standard(self, batch):
        # Custom logic here
        return super()._step_standard(batch)
```

### Accessing Components

```python
from vembed.training.data_utils import unpack_query_batch
from vembed.training.checkpoint import save_checkpoint

# Use individual components
q_batch = unpack_query_batch(batch, "t2i")
save_checkpoint("./ckpt", model, accelerator, config=config)
```

## Configuration

See parent `vembed/config.py` for full configuration options.

Key training parameters:

```yaml
# Model
model_name: openai/clip-vit-base-patch32
encoder_mode: auto  # or 'composed'

# Training
epochs: 3
batch_size: 256
lr: 5e-5
weight_decay: 0.01

# Optimization
scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0

# Checkpointing
save_steps: 500
logging_steps: 10

# Memory optimization
gradient_checkpointing: false
use_gradient_cache: false

# Knowledge distillation
teacher_model_name: null
distillation_alpha: 0.5

# LoRA
use_lora: false
lora_r: 16
```

## Performance Tips

1. **Use gradient caching** for large models:
   ```python
   config["use_gradient_cache"] = True
   ```

2. **Enable gradient checkpointing**:
   ```python
   config["gradient_checkpointing"] = True
   ```

3. **Use LoRA** for parameter-efficient training:
   ```python
   config["use_lora"] = True
   ```

4. **Increase batch size** for distributed training:
   ```bash
   accelerate launch --multi_gpu train.py --config_override batch_size=512
   ```

## Troubleshooting

### Out of Memory

Try gradient caching or checkpointing:

```python
config["use_gradient_cache"] = True
config["gradient_checkpointing"] = True
```

### Processor not found

Install missing dependencies:

```bash
pip install sentencepiece  # for SentencePiece models
pip install pillow         # for image processing
```

### DDP issues

Check `ddp_find_unused_parameters`:

```python
config["ddp_find_unused_parameters"] = True
```

---

For more details, see [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)
