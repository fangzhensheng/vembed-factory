# Data Preparation Guide

vembed-factory's `GenericRetrievalDataset` supports flexible data formats for multiple retrieval tasks.

## Data Formats

All data formats support the same retrieval modes. Specify the mode with `retrieval_mode` parameter:

```yaml
retrieval_mode: "t2i"  # Options: t2i, i2i, i2t, t2t, m2i, m2t
```

### Text-to-Image (T2I)

Retrieve images from text queries.

**JSONL Format:**
```jsonl
{"query": "red cat", "positive": "cat_red.jpg", "negatives": ["dog.jpg"]}
{"query": "blue car", "positive": "car_blue.jpg", "negatives": ["car_red.jpg"]}
```

**Column Mapping:**
```yaml
data:
  query_column: "query_text"
  positive_column: "image_path"
  negative_column: "negative_images"
```

### Image-to-Image (I2I)

Retrieve similar images.

```jsonl
{"query_image": "cat_001.jpg", "positive": "cat_001_angle2.jpg", "negatives": ["dog_001.jpg"]}
```

### Image-to-Text (I2T)

Generate or retrieve text descriptions for images.

```jsonl
{"query_image": "landscape.jpg", "positive": "sunset with mountains", "negatives": ["busy street"]}
```

### Multimodal (M2I / Composed Image Retrieval)

Combine text instructions with image queries.

```jsonl
{"query_image": "blue_shirt.jpg", "query": "change to red", "positive": "red_shirt.jpg"}
```

### Text-to-Text (T2T)

Semantic text retrieval.

```jsonl
{"query": "What is AI?", "positive": "Artificial Intelligence is...", "negatives": ["Dogs are..."]}
```

## Input File Formats

### JSONL (Recommended)

```jsonl
{"query": "...", "positive": "...", "negatives": ["..."]}
{"query": "...", "positive": "...", "negatives": ["..."]}
```

The `GenericRetrievalDataset` in `vembed/data/dataset.py` automatically parses each line.

### CSV/TSV

```csv
query,positive,negatives
"red cat","cat_red.jpg","[""dog.jpg""]"
```

**Load with column mapping:**
```python
trainer.train(
    data_path="data/train.csv",
    query_column="search_text",
    positive_column="image",
    negative_column="negatives"
)
```

### Parquet

```python
import pandas as pd

data = [
    {"query": "red cat", "positive": "cat_red.jpg", "negatives": ["dog.jpg"]},
]
df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

### HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("user/my_embedding_dataset")
trainer.train_from_dataset(dataset)
```

## Data Structure in Code

The `GenericRetrievalDataset` class in `vembed/data/dataset.py` handles parsing:

```python
# Example of how data is processed internally
batch = {
    "query": ["text query"],  # or query_image for image modes
    "positive": ["positive image/text"],
    "negatives": [["neg1", "neg2"]],
}
```

## Column Mapping

If your data uses different column names, provide a mapping:

```yaml
data:
  query_column: "search_query"
  positive_column: "matching_result"
  negative_column: "non_matching"
```

Or via Python API:

```python
trainer.train(
    data_path="data/train.jsonl",
    query_column="search_query",
    positive_column="matching_result",
    negative_column="non_matching"
)
```

## Data Validation

Check your data before training:

```python
import json
from pathlib import Path
from PIL import Image

with open("data/train.jsonl") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)

            # Validate image paths for T2I/I2I modes
            if "positive" in data and data["positive"].endswith((".jpg", ".png")):
                Image.open(data["positive"])

        except Exception as e:
            print(f"Line {i}: {e}")
```

## Data Size Recommendations

| Size | Recommendation |
|:---|:---|
| < 1K samples | Use data augmentation, consider using pre-trained embeddings |
| 1K - 10K | Good starting point for fine-tuning |
| 10K - 100K | Ideal for quality training |
| > 100K | Use gradient caching (`--use_gradient_cache`) |

## Best Practices

1. **Verify image paths exist** - Relative or absolute paths should be correct
2. **Balance negatives** - Use appropriate number of negative samples per query
3. **Data quality over quantity** - Well-labeled small dataset beats noisy large dataset
4. **Store images locally** - Don't use URLs; pre-download to local SSD
5. **Check for duplicates** - Remove duplicate query-positive pairs
6. **Set `image_root`** if images are in a specific directory:

```yaml
data:
  image_root: /data/images/
  data_path: data/train.jsonl  # Paths are relative to image_root
```

## Data Loading in Code

The `vembed/data/loading.py` module auto-detects format:

```python
# In trainer.py
dataset = load_data_from_path(
    data_path="train.jsonl",
    retrieval_mode="t2i",
    image_root="/data/images",
    query_column="query",
    positive_column="positive",
    negative_column="negatives"
)
```

## Collators

The training loop uses collators to batch data:

- **default.py**: For CLIP-style dual encoders
- **qwen.py**: For Qwen-VL models with special image handling

Collators handle:
- Tokenization with max length
- Image resizing and normalization
- Dynamic batch composition
- Negative sampling
