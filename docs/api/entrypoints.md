# Training Entrypoints

CLI entry points for training and evaluation.

## Overview

The entrypoints module provides the main functions for training, evaluation, and result generation. These are called by the CLI and can also be used programmatically for custom training workflows.

### Key Functions

| Function | Purpose |
|----------|---------|
| `train_main()` | Main training entry point with DDP support |
| `evaluate_main()` | Evaluation with full metrics |
| `evaluate_simple_main()` | Quick evaluation utility |

## Quick Start

### Train via CLI
```bash
python run.py examples/clip_train.yaml
```

### Evaluate
```bash
python run.py examples/clip_train.yaml --eval_only
```

### Multi-GPU Training
```bash
accelerate launch --multi_gpu run.py examples/clip_train.yaml
```

## Common Patterns

### Programmatic Training
```python
from vembed.entrypoints.train import train_main
import sys

# Set up arguments
sys.argv = ["train.py", "config.yaml", "--batch_size", "256"]

# Run training
train_main()
```

### Custom Training Loop
```python
from vembed.trainer import VEmbedFactoryTrainer

trainer = VEmbedFactoryTrainer("openai/clip-vit-base-patch32")
trainer.train(data_path="data.jsonl", epochs=3)
```

---

::: vembed.entrypoints.train.train_main
::: vembed.entrypoints.evaluate.evaluate_main
