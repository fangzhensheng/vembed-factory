# Base Embedding Model

Abstract base class for all embedding models.

## Overview

`BaseEmbeddingModel` defines the interface that all embedding model implementations must follow. It handles common functionality like pooling, device management, and encoding.

### Key Methods

| Method | Purpose |
|--------|---------|
| `forward()` | Forward pass returning embeddings |
| `pool()` | Pooling strategy (mean, cls, max) |
| `encode()` | Encode text or images |

## Quick Start

```python
from vembed.model.base import BaseEmbeddingModel

# Model automatically inherits from BaseEmbeddingModel
model = BaseEmbeddingModel.from_pretrained("openai/clip-vit-base-patch32")

# Get embeddings
text_emb = model.encode_text("hello")
image_emb = model.encode_image("image.jpg")
```

## Pooling Strategies

```python
model = BaseEmbeddingModel.from_pretrained(
    "model_name",
    pooling_method="mean"  # or "cls", "max"
)
```

---

::: vembed.model.base.BaseEmbeddingModel
::: vembed.model.base.disable_kv_cache
