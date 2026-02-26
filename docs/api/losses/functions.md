# Loss Functions

Collection of training objectives for contrastive learning.

## Overview

Loss functions implement various training objectives for embedding models. Each is optimized for specific scenarios:

### Available Losses

| Loss | Use Case | Formula |
|------|----------|---------|
| InfoNCE | Standard contrastive learning | Softmax over negatives |
| Triplet | Hard negative mining | Margin-based ranking |
| CoSENT | Symmetric similarity | Cosine similarity |
| ColBERT | Fine-grained retrieval | Multi-vector matching |
| MRL | Multi-scale embeddings | Layerwise supervision |

## Quick Start

```python
from vembed.losses import create_loss

# Create loss
loss_fn = create_loss(
    loss_type="infonce",
    temperature=0.07,
    scale=20
)

# Use in training
loss = loss_fn(query_emb, positive_emb, negative_embs)
loss.backward()
```

## Common Losses

### InfoNCE (Standard)
```python
loss = create_loss("infonce", temperature=0.07)
```

### Triplet Loss
```python
loss = create_loss("triplet", margin=0.3)
```

### CoSENT
```python
loss = create_loss("cosent", scale=20)
```

---

::: vembed.losses.functions.InfoNCE
::: vembed.losses.functions.TripletLoss
::: vembed.losses.functions.create_loss
