# Python API Guide

vembed-factory provides multiple Python APIs for different use cases. This guide explains the differences and when to use each.

## API Overview

| API | Complexity | Use Case |
|-----|-----------|----------|
| **Trainer** (High-level) | Low | Quick prototyping, simple workflows |
| **Modular APIs** (Mid-level) | Medium | Custom training loops, advanced control |
| **Predictor** (Inference) | Low | Encode text/images, embeddings |

---

## 1. High-Level API: Simple Training

### When to Use

- Quick prototyping
- Simple training workflows
- No custom modifications needed

### VEmbedFactoryTrainer

**Import:**

```python
from vembed.trainer import VEmbedFactoryTrainer
```

**Basic Usage:**

```python
# Initialize trainer with model
trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")

# Train
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    epochs=3,
    batch_size=32,
)
```

**Advanced Configuration:**

```python
trainer = VEmbedFactoryTrainer(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    learning_rate=1e-5,
    batch_size=64,
    scheduler_type="cosine",
    warmup_ratio=0.1,
)

trainer.train(
    data_path="data/train.jsonl",
    val_data_path="data/val.jsonl",
    output_dir="output",
    epochs=3,
)
```

**Methods:**

```python
trainer = VEmbedFactoryTrainer(model_name)

# Training
trainer.train(
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    val_data_path: str = None,
)

# Evaluation
metrics = trainer.evaluate(
    data_path="data/test.jsonl",
    batch_size=32,
)
```

**Configuration Override:**

```python
trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")

# Override via train() method
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    # Override defaults
    epochs=5,
    batch_size=64,
    learning_rate=5e-5,
    use_lora=True,
)
```

---

## 2. Inference API: Predictor

### When to Use

- Generate embeddings for text/images
- Use fine-tuned models
- Production inference

### Basic Usage

```python
from vembed.inference import VEmbedFactoryPredictor

# Initialize with trained model
predictor = VEmbedFactoryPredictor(
    model_path="output/checkpoint-epoch-3"
)

# Encode text
text_embedding = predictor.encode_text("a cat sitting on a chair")

# Encode image
image_embedding = predictor.encode_image("cat.jpg")

# Batch encoding
texts = ["a cat", "a dog", "a bird"]
embeddings = predictor.encode_text(texts, batch_size=32)
```

### Methods

```python
predictor = VEmbedFactoryPredictor(model_path)

# Text encoding
embeddings = predictor.encode_text(
    texts: Union[str, List[str]],
    batch_size: int = 32,
)  # Returns: np.ndarray of shape (n_samples, embedding_dim)

# Image encoding
embeddings = predictor.encode_image(
    images: Union[str, List[str]],
    batch_size: int = 32,
)  # Returns: np.ndarray

# Similarity search
similarities = predictor.similarity(
    query_text="a cat",
    documents=["cat.jpg", "dog.jpg", "bird.jpg"],
)  # Returns: array of similarities
```

### Configuration

```python
# Specify device
predictor = VEmbedFactoryPredictor(
    model_path="output/checkpoint-epoch-3",
    device="cuda:0",
)

# Half precision
predictor = VEmbedFactoryPredictor(
    model_path="output/checkpoint-epoch-3",
    dtype="float16",  # or "bfloat16", "float32"
)

# Pooling method
predictor = VEmbedFactoryPredictor(
    model_path="output/checkpoint-epoch-3",
    pooling_method="mean",  # or "cls", "last"
)
```

---

## 3. Mid-Level API: Modular Components

### When to Use

- Custom training loops
- Advanced control over training process
- Research and experimentation
- Integration with external frameworks

### Architecture

```
Training Pipeline:
  1. Load config      → load_and_parse_config()
  2. Build model      → build_model()
  3. Load processor   → load_processor()
  4. Build dataset    → VisualRetrievalDataset()
  5. Build dataloader → DataLoader()
  6. Build optimizer  → build_optimizer()
  7. Build loss       → LossFactory.create()
  8. Initialize distributed → Accelerator()
  9. Train            → Trainer.train()
```

### Example: Custom Training Loop

```python
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from vembed.training.config import load_and_parse_config
from vembed.training.model_builder import build_model, load_processor
from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.losses.factory import LossFactory
from vembed.training.optimizer_builder import build_optimizer

# 1. Load configuration
config = load_and_parse_config()

# 2. Initialize accelerator (handles DDP/FSDP automatically)
accelerator = Accelerator()

# 3. Build model
model = build_model(config)

# 4. Load processor
processor = load_processor(config["model_name"])

# 5. Prepare dataset
dataset = VisualRetrievalDataset(
    data_source=config["data_path"],
    processor=processor,
    mode="train",
)

# 6. Create dataloader
collator = CollatorRegistry.get("default")(processor=processor, mode="train")
dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    collate_fn=collator,
    shuffle=True,
)

# 7. Build optimizer
optimizer = build_optimizer(model, config)

# 8. Build loss
criterion = LossFactory.create(config)

# 9. Prepare for distributed training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 10. Custom training loop
model.train()
for epoch in range(config["epochs"]):
    for batch in dataloader:
        # Forward pass
        outputs = model(**batch)

        # Compute loss
        loss = criterion(outputs, batch)

        # Backward pass
        accelerator.backward(loss)

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    accelerator.print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

### Key Modules

#### Config Module

```python
from vembed.training.config import (
    load_and_parse_config,
    load_base_config,
    merge_configs,
)

# Load config from YAML + CLI overrides
config = load_and_parse_config()

# Load just defaults
defaults = load_base_config()

# Merge configs
merged = merge_configs(defaults, user_config)
```

#### Model Builder

```python
from vembed.training.model_builder import (
    build_model,
    load_processor,
    apply_lora,
    compile_model,
)

# Build model
model = build_model(config)

# Load processor
processor = load_processor("openai/clip-vit-base-patch32")

# Apply LoRA
apply_lora(model, config, accelerator)

# Compile with torch.compile (optional)
model = compile_model(model, config, accelerator)
```

#### Dataset and DataLoader

```python
from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from torch.utils.data import DataLoader

# Create dataset
dataset = VisualRetrievalDataset(
    data_source="data/train.jsonl",
    processor=processor,
    mode="train",  # or "eval"
)

# Create collator
collator = CollatorRegistry.get("default")(
    processor=processor,
    mode="train",
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator,
    shuffle=True,
)
```

#### Loss Functions

```python
from vembed.losses.factory import LossFactory

# Create loss
criterion = LossFactory.create(config)

# Or specify explicitly
criterion = LossFactory.create_loss(
    loss_type="infonce",
    config=config,
)

# Use in training
loss = criterion(embeddings, labels)
```

#### Optimizer Builder

```python
from vembed.training.optimizer_builder import (
    build_optimizer,
    build_scheduler,
)

# Create optimizer
optimizer = build_optimizer(model, config)

# Create scheduler
scheduler, warmup_steps = build_scheduler(
    optimizer, config, num_epochs, steps_per_epoch
)
```

#### Trainer Class

```python
from vembed.training.training_loop import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=dataloader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
)

# Train
trainer.train()
```

---

## 4. Practical Examples

### Example 1: Quick Prototyping

```python
from vembed.trainer import VEmbedFactoryTrainer

# One-liner training
trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")
trainer.train("data/train.jsonl", "output", epochs=3)

# Evaluate
metrics = trainer.evaluate("data/test.jsonl")
print(metrics)
```

### Example 2: Custom Training with LoRA

```python
from vembed.training.model_builder import build_model, apply_lora
from accelerate import Accelerator
from vembed.training.training_loop import Trainer

config = {
    "model_name": "Qwen/Qwen3-VL-Embedding-2B",
    "use_lora": True,
    "lora_r": 16,
}

accelerator = Accelerator()
model = build_model(config)
apply_lora(model, config, accelerator)

# ... setup optimizer, dataloader, etc ...

trainer = Trainer(model=model, optimizer=optimizer, ...)
trainer.train()
```

### Example 3: Distributed Training (FSDP)

```python
from vembed.training.config import load_and_parse_config
from vembed.training.training_loop import Trainer
from accelerate import Accelerator

config = load_and_parse_config()

accelerator = Accelerator()  # Automatically handles FSDP
model = build_model(config)

# ... setup everything ...

trainer = Trainer(model=model, ...)
trainer.train()  # Automatically uses FSDP if configured
```

### Example 4: Inference Pipeline

```python
from vembed.inference import VEmbedFactoryPredictor
import numpy as np

# Load model
predictor = VEmbedFactoryPredictor("output/checkpoint-epoch-3")

# Encode texts
queries = ["a cat", "a dog", "a bird"]
query_embs = predictor.encode_text(queries)  # (3, 512)

# Encode images
image_paths = ["cat.jpg", "dog.jpg", "bird.jpg"]
image_embs = predictor.encode_image(image_paths)  # (3, 512)

# Compute similarity
similarity = np.dot(query_embs, image_embs.T)  # (3, 3)
print(f"Query-Image similarities:\n{similarity}")
```

---

## Choosing an API

### Use Trainer (High-Level) if:
- Just want to train and evaluate
- No complex requirements
- Learning vembed-factory

```python
from vembed.trainer import VEmbedFactoryTrainer
trainer = VEmbedFactoryTrainer(model_name)
trainer.train(data_path, output_dir, epochs=3)
```

### Use Modular APIs (Mid-Level) if:
- Need custom training logic
- Want to integrate with other frameworks
- Implementing research ideas

```python
from vembed.training.model_builder import build_model
model = build_model(config)
# ... custom training ...
```

### Use Predictor (Inference) if:
- Encoding text/images
- Production inference
- Building applications

```python
from vembed.inference import VEmbedFactoryPredictor
predictor = VEmbedFactoryPredictor(model_path)
embeddings = predictor.encode_text("query")
```

---

## API Reference

See [docs/api/](../api/) for detailed API documentation:

- [Trainer API](../api/training/trainer.md)
- [Model Builder](../api/training/model_builder.md)
- [Dataset](../api/data/dataset.md)
- [Losses](../api/losses/functions.md)
- [Inference](../api/inference.md)

---

## See Also

- [Getting Started](./getting-started.md)
- [Configuration Guide](./configuration.md)
- [FSDP Training](./fsdp-training.md)
- [LoRA Fine-tuning](./lora-finetuning.md)
- [Distributed Training](./distributed-training.md)
