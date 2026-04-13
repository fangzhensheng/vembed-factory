# Training Entrypoints

CLI entry points for training and evaluation.

## Overview

The entrypoints module contains the concrete functions used by the CLI and by advanced programmatic integrations.

### Key Functions

| Function | Purpose |
|----------|---------|
| `train_entrypoint()` | Core training logic from a config dictionary |
| `main()` | CLI-compatible training entry point |

## Quick Start

### Train via CLI

```bash
vembed train examples/quickstart/clip_minimal.yaml
```

### Train via `accelerate`

```bash
accelerate launch vembed/entrypoints/train.py     --config examples/quickstart/clip_minimal.yaml
```

### Programmatic Training

```python
from vembed.entrypoints.train import train_entrypoint

config = {
    "model_name": "openai/clip-vit-base-patch32",
    "model_name_or_path": "openai/clip-vit-base-patch32",
    "data_path": "data/train.jsonl",
    "output_dir": "output/programmatic",
    "epochs": 1,
    "batch_size": 32,
}

result = train_entrypoint(config)
print(result["model_path"])
```

### High-Level Python API

```python
from vembed import Trainer

trainer = Trainer("openai/clip-vit-base-patch32", output_dir="output/clip_run")
trainer.train(data_path="data.jsonl", epochs=3)
```

## API Reference

::: vembed.entrypoints.train.train_entrypoint
::: vembed.entrypoints.train.main
