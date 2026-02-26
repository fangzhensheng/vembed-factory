# Visual Retrieval Dataset

Unified dataset class for all retrieval tasks.

## Overview

`VisualRetrievalDataset` handles loading training data from various formats (JSONL, CSV, Parquet, HuggingFace Datasets) and provides flexible column mapping for different data layouts.

### Key Features
- Multiple input formats (JSONL, CSV, Parquet, HF Datasets)
- Flexible column mapping
- Automatic image loading and caching
- Support for multiple negatives
- Validation and error checking

## Quick Start

```python
from vembed.data.dataset import VisualRetrievalDataset

# Load JSONL data
dataset = VisualRetrievalDataset(
    data_path="data/train.jsonl",
    image_root="data/images",
    columns={
        "query": "query",
        "positive": "positive",
        "negatives": "negatives"
    }
)

# Access samples
sample = dataset[0]
# Returns: {"query": "text", "positive": "path/to/image.jpg", "negatives": [...]}
```

## Supported Data Formats

### JSONL Format
```json
{"query": "text query", "positive": "pos_image.jpg", "negatives": ["neg1.jpg", "neg2.jpg"]}
```

### CSV Format
```csv
query,positive,negatives
"text","image.jpg","[neg1.jpg, neg2.jpg]"
```

### Column Mapping
```python
dataset = VisualRetrievalDataset(
    data_path="data.csv",
    columns={
        "query": "search_text",
        "positive": "target_image",
        "negatives": "hard_negatives"
    }
)
```

---

::: vembed.data.dataset.VisualRetrievalDataset
