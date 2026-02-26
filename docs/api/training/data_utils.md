# Data Utilities Module

Batch processing utilities for unpacking and concatenating training data.

**Location**: `vembed/training/data_utils.py`

**Lines**: 220

## Overview

The data_utils module handles:
- Unpacking batch data (queries, positives, negatives)
- Concatenating batches for unified models
- Extracting tensors from various model output formats
- Managing batch-level data transformations

## Key Functions

### Query Batch Unpacking

```python
def unpack_query_batch(batch: dict, mode: str) -> dict:
    """
    Extract query inputs from batch based on retrieval mode.

    Args:
        batch: Raw batch data
        mode: Retrieval mode ('t2i', 'i2i', 'i2t', 'm2i', 't2t')

    Returns:
        dict: Query inputs (pixel_values, input_ids, etc.)
    """
```

**Supported Modes**:
- `t2i` - Text-to-Image: Extract query text
- `i2i` - Image-to-Image: Extract query images
- `i2t` - Image-to-Text: Extract query images
- `m2i` - Multimodal-to-Image: Extract query text and images
- `t2t` - Text-to-Text: Extract query text

### Positive Batch Unpacking

```python
def unpack_positive_batch(batch: dict, mode: str) -> dict:
    """
    Extract positive (matching) inputs from batch.

    Similar to unpack_query_batch but extracts positive samples.
    """
```

### Negative Batch Unpacking

```python
def unpack_negative_batch(batch: dict) -> dict:
    """Extract negative (non-matching) inputs from batch."""
```

### Batch Concatenation

```python
def concat_batches(
    batches: list[dict],
    pad_id: int | None = None
) -> tuple[dict, list[int]]:
    """
    Concatenate multiple batches for unified models.

    Used when training with unified text+image models that take both
    modalities as input. Concatenates all samples and returns split points.

    Args:
        batches: List of batch dicts (query, positive, negative)
        pad_id: Padding token ID for text inputs

    Returns:
        (concatenated_batch, batch_sizes) where batch_sizes marks sample boundaries
    """
```

### Tensor Extraction

```python
def maybe_first(output):
    """
    Extract tensor from various model output formats.

    Handles:
    - Direct tensors
    - Tuple outputs (take first element)
    - NamedTuples (extract .last_hidden_state)
    - Batch outputs (take first sample)
    """
```

## Usage Example

```python
from vembed.training.data_utils import (
    unpack_query_batch,
    unpack_positive_batch,
    concat_batches,
    maybe_first,
)

# Unpack batch components
query_inputs = unpack_query_batch(batch, mode="t2i")
positive_inputs = unpack_positive_batch(batch, mode="t2i")
negative_inputs = unpack_negative_batch(batch)

# Get model embeddings
query_emb = model(query_inputs)
query_emb = maybe_first(query_emb)

# For unified models, concatenate batches
if use_unified_model:
    concat_input, batch_sizes = concat_batches(
        [query_inputs, positive_inputs],
        pad_id=processor.tokenizer.pad_token_id
    )
    embs = model(concat_input)
    embs = maybe_first(embs)
```

## Data Format Requirements

Batches must have keys matching the retrieval mode:

### Text-to-Image (T2I)
```python
batch = {
    "query": ["text query 1", "text query 2"],  # Query texts
    "positive": ["path/img1.jpg", "path/img2.jpg"],  # Positive images
    "negatives": [["neg1.jpg", "neg2.jpg"], ...],  # Negative images
}
```

### Image-to-Image (I2I)
```python
batch = {
    "query_image": ["path/img1.jpg"],  # Query images
    "positive": ["path/img_match.jpg"],  # Positive images
    "negatives": [["neg1.jpg"]],  # Negative images
}
```

### Multimodal (M2I)
```python
batch = {
    "query_image": ["path/img.jpg"],
    "query": ["modify text"],
    "positive": ["path/result.jpg"],
    "negatives": [["neg.jpg"]],
}
```

## Related Modules

- [config.md](./config.md) - Configuration (retrieval_mode)
- [training_loop.md](./training_loop.md) - Used in Trainer._train_step()
