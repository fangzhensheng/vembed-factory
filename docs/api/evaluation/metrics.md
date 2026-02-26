# Evaluation Metrics

Comprehensive metrics for retrieval task evaluation.

## Overview

The metrics module provides standard evaluation metrics for embedding models, including Recall@K, MRR, and NDCG. These metrics evaluate retrieval quality on benchmark datasets.

### Available Metrics

| Metric | Purpose | Range |
|--------|---------|-------|
| Recall@K | Fraction of queries with correct item in top-K | [0, 1] |
| MRR | Mean Reciprocal Rank | [0, 1] |
| NDCG@K | Normalized Discounted Cumulative Gain | [0, 1] |

## Quick Start

```python
from vembed.evaluation.metrics import compute_recall

# Compute Recall@1, @5, @10
query_embs = model.encode(queries)        # (N_q, D)
gallery_embs = model.encode(gallery)      # (N_g, D)

recall_at_1, recall_at_5, recall_at_10 = compute_recall(
    query_embs, gallery_embs,
    top_k=[1, 5, 10]
)
```

## Supported Evaluation Modes

### Image-to-Image (I2I)
```python
recall = compute_recall(image_query_embs, image_gallery_embs)
```

### Text-to-Image (T2I)
```python
recall = compute_recall(text_query_embs, image_gallery_embs)
```

### Image-to-Text (I2T)
```python
recall = compute_recall(image_query_embs, text_gallery_embs)
```

---

::: vembed.evaluation.metrics
