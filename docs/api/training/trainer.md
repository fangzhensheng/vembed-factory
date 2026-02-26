# VEmbedFactoryTrainer - High-Level Training API

High-level Python wrapper for quick training setups.

## Overview

`VEmbedFactoryTrainer` is a simplified training API that wraps the CLI into a Python interface. It's perfect for quick prototyping and simple training scenarios.

**Location**: `vembed/trainer.py`

**Type**: High-level API (thin wrapper around CLI)

## Quick Start

```python
from vembed.trainer import VEmbedFactoryTrainer

trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")
trainer.train(data_path="data.jsonl", output_dir="output", epochs=3)
```

## When to Use

Use `VEmbedFactoryTrainer` when you:
- Want the simplest possible API
- Are prototyping quickly
- Don't need full control over training components
- Prefer parameter-based configuration

## Alternative: Low-Level Trainer

For complete control over training, use the modular `Trainer` class from `vembed.training.training_loop`:

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model

config = load_and_parse_config()
model = build_model(config)
trainer = Trainer(model=model, optimizer=optimizer, ...)
trainer.train()
```

See [TRAINER_CLARIFICATION.md](../../TRAINER_CLARIFICATION.md) for detailed comparison.

---

## API Reference

::: vembed.trainer.VEmbedFactoryTrainer
