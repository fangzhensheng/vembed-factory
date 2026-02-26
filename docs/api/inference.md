# Inference API

High-level inference interface for embedding models.

## Overview

The inference module provides the `VEmbedFactoryPredictor` class for loading trained models and encoding text/images into embeddings. Supports batching, multi-modality, and optional dimension reduction via MRL.

### Key Features
- Simple model loading from checkpoint
- Batch encoding support
- Multi-modality support (text, image, multimodal)
- Optional MRL dimension reduction
- Automatic device management

## Quick Start

```python
from vembed.inference import VEmbedFactoryPredictor

# Load model
predictor = VEmbedFactoryPredictor("output/checkpoint-epoch-3")

# Encode text
text_emb = predictor.encode_text("a photo of a cat")  # (768,)

# Encode image
image_emb = predictor.encode_image("cat.jpg")  # (768,)

# Compute similarity
similarity = (text_emb @ image_emb.T).item()
print(f"Similarity: {similarity:.4f}")
```

## Common Use Cases

### Image Retrieval with Batch Processing
```python
import numpy as np

predictor = VEmbedFactoryPredictor("models/clip-fine-tuned")

# Encode query
query_emb = predictor.encode_image("query.jpg")

# Batch encode database
db_images = ["img1.jpg", "img2.jpg", "img3.jpg"]
db_embs = predictor.encode_image(db_images)  # (3, 768)

# Find top-k
similarities = query_emb @ db_embs.T
top_indices = np.argsort(similarities)[::-1][:10]
```

### MRL Dimension Reduction
```python
# Use 256-dim embeddings for faster search
predictor = VEmbedFactoryPredictor(
    "models/qwen3-mrl",
    mrl_dim=256
)

text_emb = predictor.encode_text("hello")  # (256,) instead of (1536,)
```

### Pooling Methods
```python
predictor = VEmbedFactoryPredictor(
    "model_path",
    pooling_method="cls"  # Options: "mean", "cls", "max"
)
```

## API Reference

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `encode_text()` | str or List[str] | (D,) or (N,D) | Encode text |
| `encode_image()` | str or List[str] | (D,) or (N,D) | Encode image |

Where D = embedding_dim, N = batch size

---

::: vembed.inference.VEmbedFactoryPredictor
