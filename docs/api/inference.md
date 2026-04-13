# Inference API

High-level inference interface for encoding text and images from trained checkpoints.

## Overview

`VEmbedModel` loads a checkpoint and exposes text and image encoding methods for retrieval use cases.

## Quick Start

```python
from vembed.inference import VEmbedModel

model = VEmbedModel("output/checkpoint-epoch-3")

text_emb = model.encode_text("a photo of a cat")
image_emb = model.encode_image("cat.jpg")
similarity = (text_emb @ image_emb.T).item()
print(f"Similarity: {similarity:.4f}")
```

For single-item inputs, the returned arrays keep a batch dimension of 1.

## Common Use Cases

### Batch Retrieval

```python
import numpy as np
from vembed.inference import VEmbedModel

model = VEmbedModel("models/clip-fine-tuned")
query_emb = model.encode_image("query.jpg")
image_embs = model.encode_image(["img1.jpg", "img2.jpg", "img3.jpg"])
similarities = query_emb @ image_embs.T
ranking = np.argsort(similarities[0])[::-1]
```

### MRL Dimension Reduction

```python
model = VEmbedModel(
    "models/qwen3-mrl",
    mrl_dim=256,
)
text_emb = model.encode_text("hello")
```

### Pooling Methods

```python
model = VEmbedModel(
    "model_path",
    pooling_method="cls",  # Common values: mean, cls, last_token, none
)
```

## API Summary

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `encode_text()` | `str` or `list[str]` | `(N, D)` | Encode text inputs |
| `encode_image()` | image path, `PIL.Image`, or list | `(N, D)` | Encode image inputs |
| `encode()` | text or image input | `(N, D)` | Generic wrapper |

`N` is the batch size. Single-item inputs still return `N=1`.

## Notes

- `encode_text()` and `encode_image()` do not accept a `batch_size` parameter.
- `VEmbedModel` does not provide a built-in `similarity()` helper.
- Set `mrl_dim` on the constructor to truncate embeddings during inference.

## API Reference

::: vembed.inference.VEmbedModel
