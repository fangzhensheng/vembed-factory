# Trainer

High-level training API.

## Overview

High-level Python interface for model training.

## Quick Start

```python
from vembed import Trainer

trainer = Trainer("openai/clip-vit-base-patch32")
trainer.train(data_path="data.jsonl", output_dir="output")
```

---

::: vembed.trainer.VEmbedFactoryTrainer
