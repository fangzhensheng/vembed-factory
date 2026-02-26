# Data Loading

Utilities for loading data from multiple formats.

## Overview

The loading module handles data import from JSONL, CSV, Parquet, and HuggingFace Datasets with flexible column mapping and validation.

## Quick Start

```python
from vembed.data.loading import load_dataset

# Load JSONL
dataset = load_dataset(
    data_path="data/train.jsonl",
    format="jsonl"
)

# Load from HuggingFace
dataset = load_dataset(
    dataset_name="some/hf-dataset",
    split="train"
)
```

---

::: vembed.data.loading
