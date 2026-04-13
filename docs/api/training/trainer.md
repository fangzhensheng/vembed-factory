# VEmbedTrainer - High-Level Training API

High-level Python wrapper for launching training from code.

## Overview

`vembed.trainer.VEmbedTrainer` is the public high-level trainer implementation. The package-level alias `from vembed import Trainer` points to this same class.

**Location**: `vembed/trainer.py`

## Quick Start

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

## When to Use

Use `VEmbedTrainer` when you:

- Want a compact Python API
- Prefer constructor plus method parameters over building the whole training stack yourself
- Need to pass a few advanced config values with `**kwargs`

## API Notes

- Set `output_dir` on the constructor.
- `train()` returns a dictionary containing `output_dir` and `model_path`.
- `train()` forwards extra keyword arguments into the merged config.
- This class does not expose `evaluate()`.

## Alternative: Low-Level Trainer

For full control over the training loop, use `vembed.training.Trainer`.

```python
from vembed.training import Trainer, load_and_parse_config
from vembed.training.model_builder import build_model

config = load_and_parse_config()
model = build_model(config)
trainer = Trainer(model=model, optimizer=optimizer, dataloader=dataloader, criterion=criterion, accelerator=accelerator, config=config)
trainer.train()
```

## API Reference

::: vembed.trainer.VEmbedTrainer
