# Getting Started with vembed-factory

vembed-factory is a training framework optimized for visual and multimodal embeddings. This guide shows you how to train your first model in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# Install with uv (recommended)
uv sync
source .venv/bin/activate

# Or with pip
pip install -e ".[all]"
```

## Three Ways to Train

vembed-factory supports multiple training approaches:

| Method | Use Case | Complexity |
|--------|----------|-----------|
| **CLI** (`vembed train`) | Production, batch training | Low |
| **Python API - Simple** (`VEmbedFactoryTrainer`) | Prototyping, quick experiments | Low |
| **Python API - Advanced** (`Trainer` from `vembed.training`) | Research, customization | Medium |

Choose based on your needs. All three are 100% compatible and use the same training core.

---

## Your First Training

### 1. Prepare Data

Create `data/train.jsonl` with retrieval pairs:

```jsonl
{"query": "a red cat", "positive": "cat_red.jpg", "negatives": ["dog.jpg", "cat_blue.jpg"]}
{"query": "a dog running", "positive": "dog_running.jpg", "negatives": ["dog_sitting.jpg"]}
```

Supported formats:
- **Text-to-Image (T2I)**: `{"query": "...", "positive": "image.jpg", "negatives": [...]}`
- **Image-to-Image (I2I)**: `{"query_image": "...", "positive": "...", "negatives": [...]}`
- **Image-to-Text (I2T)**: `{"query_image": "...", "positive": "text"}`
- **Multimodal (M2I)**: `{"query_image": "...", "query": "...", "positive": "..."}`
- **Text-to-Text (T2T)**: `{"query": "...", "positive": "..."}`

### 2. Train with Python API (Simple)

```python
from vembed.trainer import VEmbedFactoryTrainer

# Initialize with any HuggingFace model
trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")

# Train in 3 lines
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    epochs=3
)
```

This uses the high-level API wrapper that delegates to CLI internally. Perfect for quick prototyping.

### 3. Train with CLI

```bash
# Simple training
accelerate launch vembed/entrypoints/train.py --config examples/clip_train.yaml

# Override parameters
accelerate launch vembed/entrypoints/train.py \
    --config examples/clip_train.yaml \
    --batch_size 64 \
    --learning_rate 1e-5

# Via CLI tool
vembed train --config examples/clip_train.yaml --batch_size 64
```

This is the recommended approach for production training with full support for distributed training.

### 4. Train with Python API (Advanced)

For complete control over training, use the modular training components:

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model
from vembed.training.optimizer_builder import build_optimizer
from accelerate import Accelerator

# Load and parse configuration
config = load_and_parse_config()

# Build components
accelerator = Accelerator()
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

This gives you full control to customize the training loop. See [vembed/training/README.md](../../vembed/training/README.md) for detailed API documentation.

### 5. Use the Trained Model

```python
from vembed import Predictor

# Load the model
predictor = Predictor("output/checkpoint-epoch-3")

# Encode text
text_emb = predictor.encode_text("a red cat")

# Encode image
image_emb = predictor.encode_image("cat_red.jpg")

# Compute similarity
score = (text_emb * image_emb).sum()
print(f"Similarity: {score:.4f}")
```

## What's Next?

- **[Data Preparation Guide](data-preparation.md)** - Learn about data formats and preprocessing
- **[Configuration Guide](configuration.md)** - Explore training options
- **[Advanced Training](training-advanced.md)** - Gradient caching, LoRA, MRL, etc.

## Supported Models

vembed-factory works with any HuggingFace model, with built-in optimization for:

- **CLIP / SigLIP**: Fast dual-encoders
- **Qwen3-VL-Embedding**: State-of-the-art multimodal models
- **ColPali**: Document and fine-grained retrieval
- **Custom models**: Mix any text + image encoder

## About the Training Module Refactoring

The training module has been reorganized into 8 specialized components to improve maintainability and testability while maintaining 100% backward compatibility:

| Module | Purpose | Lines |
|--------|---------|-------|
| **config.py** | Configuration loading & parsing | 60 |
| **data_utils.py** | Batch unpacking & concatenation | 220 |
| **optimizer_builder.py** | Optimizer & scheduler creation | 110 |
| **model_builder.py** | Model initialization with optimizations | 200 |
| **checkpoint.py** | Checkpoint saving & management | 60 |
| **evaluator.py** | Validation evaluation loop | 130 |
| **training_loop.py** | Core `Trainer` class | 490 |
| **__init__.py** | Public API exports | - |

**Total**: 1,265 lines across 8 focused modules (vs. original 790 lines in single file)

### Why This Refactoring?

✅ **Better maintainability** - Each module has single responsibility
✅ **Easier testing** - Each component can be tested independently
✅ **Better reusability** - Import and use modules independently in Python
✅ **100% backward compatible** - All CLI commands still work
✅ **New flexibility** - Can now compose training components programmatically

### Migration Path

- **If using CLI**: No changes needed! All commands work exactly as before.
- **If using `VEmbedFactoryTrainer`**: No changes needed! Still works as before.
- **If want full control**: New! Use `vembed.training.Trainer` directly for complete customization.

See [REFACTORING_SUMMARY.md](../../REFACTORING_SUMMARY.md) for complete details.

---

## Common Issues

**Q: Out of memory?**
```bash
accelerate launch vembed/entrypoints/train.py config.yaml --batch_size 8 --config_override use_gradient_cache=true
```

**Q: How to log to W&B?**
```bash
wandb login
accelerate launch vembed/entrypoints/train.py config.yaml --config_override report_to=wandb
```

**Q: What's the difference between VEmbedFactoryTrainer and the new Trainer?**

See [TRAINER_CLARIFICATION.md](../../TRAINER_CLARIFICATION.md) for a detailed comparison.

See [Troubleshooting](../troubleshooting.md) for more help.
