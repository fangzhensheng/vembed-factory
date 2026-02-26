# Training Loop Module

Core Trainer class for orchestrating the complete training process.

**Location**: `vembed/training/training_loop.py`

**Lines**: 490

## Overview

The training_loop module provides the main `Trainer` class that orchestrates:
- Complete training loop across epochs
- Forward passes (with optional gradient caching)
- Backward passes and gradient accumulation
- Knowledge distillation
- Checkpoint saving during training
- Validation evaluation

## Key Class

### `Trainer`

```python
class Trainer:
    """
    Core trainer for visual retrieval models.

    Handles complete training orchestration including:
    - Gradient caching for memory efficiency
    - Knowledge distillation
    - Distributed training (DDP)
    - Mixed precision training
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        accelerator,
        config: dict,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        evaluator: Evaluator | None = None,
        teacher_model: torch.nn.Module | None = None,
        distillation_loss_fn: Callable | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Student model to train
            optimizer: AdamW optimizer
            dataloader: Training DataLoader
            criterion: Loss function
            accelerator: Accelerate accelerator
            config: Training configuration dict
            scheduler: LR scheduler (optional)
            evaluator: Evaluator for validation (optional)
            teacher_model: Teacher model for distillation (optional)
            distillation_loss_fn: Distillation loss function (optional)
        """

    def train(self) -> None:
        """
        Run complete training loop.

        Executes:
        1. Training for num_epochs epochs
        2. Validation at configured intervals
        3. Checkpoint saving
        4. Metric logging

        Raises:
            RuntimeError: If training fails
        """
```

## Usage Example

### Basic Training

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer
from accelerate import Accelerator

# Setup
config = load_and_parse_config()
accelerator = Accelerator()

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
)

# Train
trainer.train()
```

### With Validation

```python
from vembed.training import Trainer
from vembed.training.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(model, criterion, accelerator)

# Create trainer with validation
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
    evaluator=evaluator,
)

trainer.train()
```

### With Knowledge Distillation

```python
from vembed.training import Trainer
from vembed.training.model_builder import build_teacher_model

# Build student and teacher
student_model = build_model(config)
teacher_model = build_teacher_model(config)

# Freeze teacher
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# Create distillation loss
from vembed.losses.functions.distillation import DistillationLoss
distill_fn = DistillationLoss(temperature=4.0)

# Train
trainer = Trainer(
    model=student_model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    teacher_model=teacher_model,
    distillation_loss_fn=distill_fn,
)

trainer.train()
```

### Custom Training with Inheritance

```python
from vembed.training import Trainer

class CustomTrainer(Trainer):
    """Custom trainer with custom training step."""

    def _step_standard(self, batch):
        """Override standard training step."""
        # Custom logic here
        query_emb = self.model(batch['query'])

        # Your custom processing
        loss = self.criterion(query_emb, batch['labels'])

        return loss, {"custom_metric": loss.item()}

# Use custom trainer
trainer = CustomTrainer(...)
trainer.train()
```

## Training Process

### Main Training Loop

```
for epoch in range(num_epochs):
    model.train()

    for step, batch in enumerate(dataloader):
        # 1. Forward pass (with optional gradient cache)
        loss = _train_step(batch)

        # 2. Backward pass + optimizer step
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # 3. Logging
        if step % logging_steps == 0:
            _log_step(loss, step, epoch)

        # 4. Checkpoint saving
        if save_steps > 0 and step % save_steps == 0:
            _save_checkpoint(step)

    # 5. Epoch-end validation
    if evaluator:
        metrics = evaluator.evaluate(val_dataloader)

    # 6. Epoch-end checkpoint
    _save_checkpoint_epoch(epoch)
```

## Key Methods

### `_train_step(batch) -> float`

Executes single training step with:
- Optional gradient caching
- Optional knowledge distillation
- Loss computation

### `_step_standard(batch) -> tuple[Tensor, dict]`

Standard forward pass for:
- Text-to-Image training
- Image-to-Image training
- Other retrieval modes

Returns loss and metrics dict.

### `_apply_distillation(batch) -> Tensor`

Applies knowledge distillation:
- Computes teacher embeddings
- Computes student embeddings
- Computes distillation loss

## Configuration Reference

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 3 | Number of training epochs |
| `save_steps` | int | 0 | Save checkpoint every N steps (0=off) |
| `logging_steps` | int | 10 | Log metrics every N steps |
| `eval_steps` | int | 0 | Evaluate every N steps (0=epoch-end only) |
| `max_grad_norm` | float | 1.0 | Gradient clipping norm (0 to disable) |

### Gradient Cache Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gradient_cache` | bool | true | Enable gradient caching |
| `gradient_cache_chunk_size` | int | 32 | Chunk size for gradient cache |

### Distillation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_model_name` | str | None | Teacher model for distillation |
| `distillation_weight` | float | 0.5 | Weight of distillation loss |
| `distillation_temperature` | float | 4.0 | Temperature for softening |

## Performance Tips

1. **Use Gradient Cache** for large batches
2. **Enable Gradient Checkpointing** if OOM
3. **Use LoRA** for parameter-efficient training
4. **Use Torch Compile** on PyTorch 2.0+ for speedup
5. **Reduce eval_steps** for faster training

## Debugging

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile training step

```python
import cProfile

trainer = Trainer(...)

cProfile.run("trainer.train()")
```

## Related Modules

- [config.md](./config.md) - Configuration
- [model_builder.md](./model_builder.md) - Model initialization
- [optimizer_builder.md](./optimizer_builder.md) - Optimizer/scheduler
- [data_utils.md](./data_utils.md) - Batch processing
- [checkpoint.md](./checkpoint.md) - Checkpoint saving
- [evaluator.md](./evaluator.md) - Validation
