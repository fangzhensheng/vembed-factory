# Data Loading and Configuration Guide

This guide covers the enhanced data loading features in vembed-factory, including custom column mapping, data validation, and performance optimization.

## Table of Contents

1. [Column Mapping](#column-mapping)
2. [Data Validation](#data-validation)
3. [Performance Optimization](#performance-optimization)
4. [Image Caching](#image-caching)
5. [Troubleshooting](#troubleshooting)

---

## Column Mapping

### What is Column Mapping?

Column mapping allows you to use datasets with **non-standard column names**. By default, vembed-factory expects columns named `query`, `positive`, and `negatives`. If your dataset uses different names (like `text`, `image`, `hard_negs`), you can map them.

### How to Use

Add a `column_mapping` section to your YAML config:

```yaml
column_mapping:
  query: "text_query"          # Maps "text_query" column to "query"
  positive: "image_path"       # Maps "image_path" column to "positive"
  negatives: "similar_images"  # Maps "similar_images" column to "negatives"
  query_image: "source_image"  # For image-to-image retrieval (optional)
```

### Standard Column Aliases

vembed-factory automatically tries these common aliases if explicit mapping is not provided:

**For "query" field:**
- query, caption, text, question, instruction, prompt

**For "positive" field:**
- positive, image, answer, content, document, paragraph

**For "negatives" field:**
- negatives, negative_samples, hard_negatives, distractors

**For "query_image" field:**
- query_image, source_image

### Example: Using Column Mapping

```yaml
# Example 1: Text-to-Image with custom names
column_mapping:
  query: "caption"      # Dataset uses "caption" instead of "query"
  positive: "image_id"  # Dataset uses "image_id" instead of "positive"

# Example 2: Image-to-Image with custom names
column_mapping:
  query_image: "query_img"
  positive: "target_img"
  negatives: "neg_imgs"
```

See `examples/example_custom_columns.yaml` for a complete working example.

---

## Data Validation

### What is Data Validation?

Data validation checks your dataset before training to ensure:
- All required columns are present
- Data quality and distribution
- Image file paths are accessible
- No missing or malformed records

This helps catch configuration issues early.

### How to Use

Set `validate_data: true` in your config:

```yaml
validate_data: true
skip_invalid_records: true  # Automatically skip incomplete records
```

### Validation Output

When enabled, you'll see a report like:

```
============================================================
📊 DATASET VALIDATION REPORT
============================================================

📈 Record Statistics:
  Total records: 120000
  Analyzed sample: 100

📝 Text Statistics:
  Min length: 5
  Max length: 512
  Avg length: 87.3
  Missing: 0/100

🖼️  Image Statistics:
  Image samples: 95/100 (95.0%)
  Has query images: false

❌ Negative Statistics:
  Negative samples: 100/100 (100.0%)

✅ Required Fields:
  ✓ query
  ✓ positive

============================================================
```

### Common Issues

**Issue: "Missing fields" warning**
- Check your `column_mapping` configuration
- Ensure column names match your dataset exactly
- Use `validate_data: true` to verify

**Issue: "Image samples: 0%"**
- Your "positive" column might contain text, not image paths
- For text-to-text mode, this is expected
- For visual retrieval (t2i/i2i), image paths should contain file extensions (.jpg, .png, etc.)

---

## Performance Optimization

### Data Loader Parameters

Control training speed with these parameters:

```yaml
# Number of parallel workers for data loading
# Set to 0 for debugging, 4-8 for production
num_workers: 4

# Pin GPU memory for faster transfer (requires ~4GB extra RAM)
pin_memory: true

# Number of batches to prefetch per worker (higher = faster, uses more RAM)
prefetch_factor: 2

# Keep workers alive between epochs (faster than restarting workers)
persistent_workers: true
```

### Recommended Configurations

**For single GPU (debugging):**
```yaml
num_workers: 0           # Disable workers for easier debugging
pin_memory: false
```

**For multi-GPU (production):**
```yaml
num_workers: 8           # Increase workers
pin_memory: true         # Pin memory for faster GPU transfer
prefetch_factor: 3       # More aggressive prefetching
persistent_workers: true
```

**For large datasets (>100K images):**
```yaml
num_workers: 12
pin_memory: true
prefetch_factor: 4
persistent_workers: true
```

### Memory vs Speed Trade-off

- **More workers/prefetch** = Faster training, higher RAM usage
- **Fewer workers** = Slower training, lower RAM usage
- **pin_memory** = Faster GPU transfer, requires ~4GB extra CPU RAM

---

## Image Caching

### What is Image Caching?

Image caching loads and stores images in GPU memory during the first epoch. This dramatically speeds up multi-epoch training for small-to-medium datasets (<50K images).

### How to Use

Enable in your config:

```yaml
enable_image_cache: true
epochs: 20  # Effective for multiple epochs
```

### When to Use

✅ **Good for:**
- Small datasets (< 50K images)
- Multi-epoch training
- Limited I/O bandwidth

❌ **Bad for:**
- Large datasets (> 500K images) - will run out of memory
- One-shot training (no benefit)
- Streaming data sources

### Memory Estimate

Image cache size ≈ `(num_images × avg_image_size) / compression_ratio`

For 10K images with avg 2MB:
- Without cache: 0 MB
- With cache: ~20 GB

Use this formula to estimate if your GPU has enough memory.

---

## Complete Configuration Example

```yaml
# Model
model_name: facebook/dinov2-base
encoder_mode: dinov2
use_lora: true

# Custom columns (only if needed)
column_mapping:
  query_image: "query_img"
  positive: "target_img"

# Data
data_path: data/train.jsonl
val_data_path: data/val.jsonl
image_root: data/images

# Validation
validate_data: true
skip_invalid_records: true

# Performance
num_workers: 8
pin_memory: true
prefetch_factor: 2
persistent_workers: true
enable_image_cache: false  # Disable for large datasets

# Training
batch_size: 128
epochs: 20
learning_rate: 1.0e-04
```

---

## Troubleshooting

### Q: "RuntimeError: num_workers > 0 and persistent_workers=True not compatible"

**A:** This can happen with certain data types (e.g., HuggingFace Datasets). Set:
```yaml
num_workers: 0  # or persistent_workers: false
```

### Q: "OOM: out of memory" with enable_image_cache

**A:** Your dataset is too large for GPU memory. Disable caching:
```yaml
enable_image_cache: false
```

### Q: "FileNotFoundError: image not found"

**A:** Check your `image_root` and file paths:
1. Verify `image_root` points to correct directory
2. Ensure image paths in data are relative or absolute correctly
3. Use `validate_data: true` to catch these early

### Q: Training is slow despite multi-worker setup

**A:** Try increasing `prefetch_factor` or `num_workers`:
```yaml
num_workers: 12        # Increase workers
prefetch_factor: 4     # More aggressive prefetch
persistent_workers: true
```

### Q: How do I know which columns my dataset has?

**A:** Set `validate_data: true` to run validation, which will report detected columns and suggest mappings.

---

## Related Documentation

- [Model Configuration](./model_configuration.md)
- [Training Guide](./training_guide.md)
- [Loss Functions](./loss_functions.md)

## Further Reading

- PyTorch DataLoader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- Accelerate distributed training: https://huggingface.co/docs/accelerate/
