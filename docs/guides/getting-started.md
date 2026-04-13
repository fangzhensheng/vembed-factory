# Getting Started with vembed-factory

vembed-factory supports three training entry points that all lead into the same training core.

## Installation

```bash
# Clone the repository
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# Recommended: uv
uv sync
source .venv/bin/activate

# Or pip
pip install -e ".[all]"
```

## Three Ways to Train

| Method | Use Case | Complexity |
|--------|----------|-----------|
| **CLI** (`vembed train`) | Reproducible runs and production workflows | Low |
| **Python API - Simple** (`Trainer`) | Prototyping and notebooks | Low |
| **Python API - Advanced** (`vembed.training.Trainer`) | Research and custom orchestration | Medium |

## Your First Training

### 1. Prepare Data

Create `data/train.jsonl` with retrieval pairs:

```jsonl
{"query": "a red cat", "positive": "cat_red.jpg", "negatives": ["dog.jpg", "cat_blue.jpg"]}
{"query": "a dog running", "positive": "dog_running.jpg", "negatives": ["dog_sitting.jpg"]}
```

### 2. Train with the High-Level Python API

```python
from vembed import Trainer

trainer = Trainer(
    "openai/clip-vit-base-patch32",
    output_dir="output/clip_run",
)

result = trainer.train(
    data_path="data/train.jsonl",
    image_root="data/images",
    epochs=3,
)

print(result["model_path"])
```

This high-level API calls the core training entrypoint directly and is ideal for quick experiments.

### 3. Train with the CLI

```bash
# Quick start - minimal CLIP configuration
vembed train examples/quickstart/clip_minimal.yaml

# Quick start - minimal Qwen3-VL configuration
vembed train examples/quickstart/qwen3_vl_minimal.yaml

# Override parameters from CLI
vembed train examples/quickstart/clip_minimal.yaml \
    --data_path data/train.jsonl \
    --image_root data/images \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --epochs 3

# Distributed training with FSDP
vembed train examples/distributed/qwen3_vl_8b_fsdp.yaml
```

### 4. Train with the Low-Level Modules

```python
from accelerate import Accelerator

from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer

config = load_and_parse_config()
accelerator = Accelerator()
model = build_model(config)
optimizer = build_optimizer(model, config)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataloader=train_loader,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
    scheduler=scheduler,
)
trainer.train()
```

### 5. Use the Trained Model

```python
from vembed import VEmbedModel

model = VEmbedModel("output/checkpoint-epoch-3")
text_emb = model.encode_text("a red cat")
image_emb = model.encode_image("cat_red.jpg")
score = (text_emb @ image_emb.T).item()
print(f"Similarity: {score:.4f}")
```

## What Changed in the Python API

- Use `Trainer` or `VEmbedTrainer` for the high-level Python interface.
- Set `output_dir` in the constructor, not in `train()`.
- The high-level trainer returns a result dictionary and does not provide `evaluate()`.

## What Next

- [Data Preparation Guide](data-preparation.md)
- [Configuration Guide](configuration.md)
- [Python API Guide](python-api.md)
- [Distributed Training Guide](distributed-training.md)
- [LoRA Fine-tuning](lora-finetuning.md)
- [Monitoring](monitoring.md)

## Common Issues

**Q: Out of memory?**

```bash
vembed train examples/quickstart/qwen3_vl_minimal.yaml \
    --batch_size 8 \
    --use_gradient_cache \
    --gradient_checkpointing
```

**Q: How to log to W&B?**

```bash
wandb login
vembed train examples/quickstart/clip_minimal.yaml --report_to wandb
```

**Q: Which Python trainer should I choose?**

- Use `Trainer` for simple Python calls.
- Use `vembed.training.Trainer` for full control.
