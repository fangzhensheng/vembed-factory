# Collator Selection & Configuration

## Overview

Data collators are responsible for batching and processing raw data into model inputs. The collator selection and configuration works as follows:

```
Configuration (YAML)
    ↓
encoder_mode → Selects Collator Class
retrieval_mode → Passes to Collator for data processing
    ↓
Collator receives (processor, mode, retrieval_mode, ...)
    ↓
Collator.__call__(batch) → Returns model inputs
```

## Collator Selection by encoder_mode

The `encoder_mode` parameter determines which Collator class is used:

| encoder_mode | Collator Class | Models | Data |
|---|---|---|---|
| `dino`, `dinov2`, `dinov3`, `vit`, `mae` | VITFamilyCollator | DINOv2/v3, ViT, MAE | Images only |
| `clip`, `siglip` | CLIPFamilyCollator | CLIP, SigLIP | Text + Images |
| `bert`, `bge`, `e5`, `qwen` | BERTFamilyCollator | BERT, BGE, E5 | Text only |
| `qwen-vl`, `llava` | VLMRetrievalCollator | Qwen-VL, LLaVA | Text + Images (via chat) |
| (auto/not specified) | CLIPFamilyCollator | Default | Text + Images |

## Data Processing by retrieval_mode

The `retrieval_mode` parameter is passed to the collator and can be used to adapt data processing:

| Mode | Query | Corpus | Use Case |
|---|---|---|---|
| `t2i` | Text | Images | Text-to-image search |
| `i2i` | Images | Images | Image-to-image search |
| `t2t` | Text | Text | Text-to-text search |
| `i2t` | Images | Text | Image-to-text search |
| `m2i` | Multi-modal | Images | Multi-modal search |
| `m2t` | Multi-modal | Text | Multi-modal search |

## Configuration Examples

### Example 1: Image-to-Image Retrieval (DINO)

```yaml
encoder_mode: dino        # → VITFamilyCollator
retrieval_mode: "i2i"     # → Image query + image corpus
model_name: facebook/dinov2-base
data_path: images_train.jsonl
```

**Expected data format**:
```json
{
  "query_image": "path/to/query.jpg",
  "positive": "path/to/match.jpg",
  "negatives": ["path/to/neg1.jpg", "path/to/neg2.jpg"]
}
```

### Example 2: Text-to-Image Retrieval (CLIP)

```yaml
encoder_mode: clip        # → CLIPFamilyCollator
retrieval_mode: "t2i"     # → Text query + image corpus
model_name_or_path: openai/clip-vit-base-patch32
data_path: t2i_train.jsonl
image_root: images/
```

**Expected data format**:
```json
{
  "query": "a dog on grass",
  "positive": "dog.jpg",
  "negatives": ["cat.jpg", "bird.jpg"]
}
```

### Example 3: Text-to-Text Retrieval (BERT)

```yaml
encoder_mode: bert        # → BERTFamilyCollator
retrieval_mode: "t2t"     # → Text query + text corpus
model_name: bert-base-uncased
data_path: t2t_train.jsonl
```

**Expected data format**:
```json
{
  "query": "How to train a model?",
  "positive": "Model training involves...",
  "negatives": ["Random text 1", "Random text 2"]
}
```

### Example 4: Multi-Modal Retrieval (Qwen-VL)

```yaml
encoder_mode: qwen-vl     # → VLMRetrievalCollator
retrieval_mode: "m2i"     # → Multi-modal query → images
model_name: Qwen/Qwen2-VL-7B-Instruct
data_path: m2i_train.jsonl
image_root: images/
```

## Implementation Notes

### Field Auto-Detection

Collators use field auto-detection to support flexible data formats:

```python
# In BaseRetrievalCollator._detect_fields():
{
    "has_query_text": any(item.get("query_text")),
    "has_pos_text": any(item.get("pos_text")),
    "has_pos_image": any(item.get("pos_image")),
    "has_query_image": any(item.get("query_image")),
}
```

This means CLIPFamilyCollator can adapt to various retrieval modes:
- If only `query_text` and `pos_image` → t2i mode
- If only `query_image` and `pos_image` → i2i mode
- If only `query_text` and `pos_text` → t2t mode

### retrieval_mode Usage

`retrieval_mode` is available to collators via `self.retrieval_mode` for explicit logic if needed:

```python
class MyCollator(BaseRetrievalCollator):
    def __call__(self, batch):
        if self.retrieval_mode == "t2i":
            # Explicit t2i handling
        elif self.retrieval_mode == "m2i":
            # Explicit m2i handling
```

## Current Status

- ✅ `encoder_mode` fully determines collator selection
- ✅ `retrieval_mode` passed to all collators
- ✅ Field auto-detection enables flexible data formats
- ✅ All 20+ example configs define both parameters
- 🔄 Collators can optionally implement explicit `retrieval_mode` logic
