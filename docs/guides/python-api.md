# Python API Guide

vembed-factory exposes three Python API layers aimed at different levels of control.

## API Overview

| API | Complexity | Best For |
|-----|-----------|----------|
| **Trainer** (high-level) | Low | Quick experiments and simple workflows |
| **Training Modules** (`vembed.training`) | Medium | Custom training pipelines and research code |
| **VEmbedModel** (inference) | Low | Encoding text and images for retrieval |

## 1. High-Level Training API

### Public Entry Points

```python
from vembed import Trainer
from vembed.trainer import VEmbedTrainer
```

`Trainer` is the public alias of `VEmbedTrainer`.

### When to Use

- Quick prototyping
- Running training from Python without going through the CLI
- Passing a few configuration overrides from code

### Basic Usage

```python
from vembed import Trainer

trainer = Trainer(
    "openai/clip-vit-base-patch32",
    output_dir="output/clip_run",
)

result = trainer.train(
    data_path="data/train.jsonl",
    epochs=3,
    batch_size=64,
    learning_rate=5e-5,
    image_root="data/images",
)

print(result["output_dir"])
print(result["model_path"])
```

### Constructor Parameters

```python
trainer = Trainer(
    model_name="openai/clip-vit-base-patch32",
    mode="auto",
    output_dir="output",
    use_gpu=True,
    loss_type="infonce",
    collator_type=None,
)
```

### `train()` Parameters

```python
result = trainer.train(
    data_path="data/train.jsonl",
    val_data_path="data/val.jsonl",
    epochs=3,
    batch_size=64,
    learning_rate=5e-5,
    use_gradient_cache=True,
    use_mrl=False,
    mrl_dims=[768, 512, 256],
    retrieval_mode="t2i",
    encoder_mode="auto",
    text_model_name=None,
    image_model_name=None,
    save_steps=0,
    use_lora=True,
    report_to=None,
    attn_implementation=None,
    torch_dtype=None,
    gradient_checkpointing=False,
    image_root=None,
)
```

### Advanced Overrides via `**kwargs`

Extra keyword arguments are forwarded into the merged training config. This is the easiest way to pass advanced settings such as distillation options.

```python
trainer = Trainer(
    "openai/clip-vit-base-patch32",
    output_dir="output/distill_run",
)

trainer.train(
    data_path="data/train.jsonl",
    val_data_path="data/val.jsonl",
    use_lora=True,
    teacher_model_name="openai/clip-vit-large-patch14",
    distillation_alpha=0.5,
    distillation_temperature=2.0,
)
```

### Notes

- Set `output_dir` on the constructor, not on `train()`.
- `train()` returns a dictionary with `output_dir` and `model_path`.
- The high-level trainer does not provide a `trainer.evaluate()` method. Use the CLI evaluation entrypoints or lower-level modules if you need a custom evaluation loop.

## 2. Inference API: `VEmbedModel`

### When to Use

- Load a trained checkpoint
- Encode text or images into embeddings
- Build retrieval applications from exported checkpoints

### Basic Usage

```python
from vembed import VEmbedModel
import numpy as np

model = VEmbedModel("output/checkpoint-epoch-3")

query_embeddings = model.encode_text(["a cat", "a dog"])
image_embeddings = model.encode_image(["cat.jpg", "dog.jpg"])

similarity = query_embeddings @ image_embeddings.T
print(np.asarray(similarity))
```

### Constructor Parameters

```python
model = VEmbedModel(
    model_path="output/checkpoint-epoch-3",
    device="cuda",
    encoder_mode="auto",
    text_model_name=None,
    image_model_name=None,
    pooling_method="mean",
    mrl_dim=None,
)
```

### Supported Encoding Methods

```python
text_embeddings = model.encode_text("a photo of a cat")
text_batch_embeddings = model.encode_text(["a cat", "a dog"])

image_embeddings = model.encode_image("cat.jpg")
image_batch_embeddings = model.encode_image(["cat.jpg", "dog.jpg"])

embeddings = model.encode("a cat")
image_embeddings = model.encode("cat.jpg", is_image=True)
```

### Important Behavior

- `encode_text()` and `encode_image()` do not accept a `batch_size` argument.
- Single-item inputs are internally converted to a batch of size 1, so the returned shape is `(1, D)` rather than `(D,)`.
- There is no built-in `similarity()` helper; compute similarity yourself with NumPy or PyTorch.
- Use `mrl_dim` on the constructor to truncate embeddings at inference time.

### Pooling Methods

Common values are:

- `mean`
- `cls`
- `last_token`
- `none` for token-level outputs such as ColBERT-style late interaction

## 3. Low-Level Training Modules

### When to Use

- Custom training loops
- Research code
- Fine-grained control over model creation, optimizer setup, and evaluation

### Minimal Example

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader

from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer, build_scheduler

config = load_and_parse_config()
accelerator = Accelerator()
model = build_model(config)
optimizer = build_optimizer(model, config)
scheduler, _ = build_scheduler(optimizer, config, num_epochs=3, steps_per_epoch=100)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=DataLoader(...),
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
)
trainer.train()
```

## Choosing an API

### Use `Trainer` if

- You want the simplest Python training interface
- You are okay with the default training pipeline
- You only need light config overrides

### Use `vembed.training.Trainer` if

- You need control over optimizers, schedulers, datasets, or evaluators
- You want to integrate vembed-factory into another training framework
- You are building research code

### Use `VEmbedModel` if

- You only need inference and retrieval embeddings
- You are serving or evaluating a trained checkpoint
- You want to compute similarity externally

## See Also

- [Getting Started](./getting-started.md)
- [Configuration Guide](./configuration.md)
- [Monitoring](./monitoring.md)
- [Training API Reference](../api/training/trainer.md)
- [Inference API Reference](../api/inference.md)
