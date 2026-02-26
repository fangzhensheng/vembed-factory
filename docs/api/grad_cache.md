# Gradient Cache Library

Memory-efficient training for large batch sizes.

## Overview

Gradient caching enables training with very large batch sizes (512+) on limited VRAM by chunking forward passes and deferring gradient synchronization. Combined with gradient checkpointing and LoRA, it can reduce memory usage by 40-50% with minimal speed impact.

### How It Works

```
For each batch:
  1. Forward Pass (Chunk 1) [no_grad]     → Cache gradients
  2. Forward Pass (Chunk 2) [no_grad]     → Cache gradients
  3. Backward Pass (Chunk 1) [with_grad]
  4. Backward Pass (Chunk 2) [with_grad]
  5. Optimizer.step()
```

### Memory Savings
- **Effective Batch Size**: 2-4x larger than without gradient cache
- **VRAM Savings**: 30-50% reduction
- **Speed Trade-off**: ~10-20% slower due to recomputation

## Quick Start

### Configuration
```yaml
use_gradient_cache: true
gradient_cache_chunk_size: 64
batch_size: 128  # Total batch, divided into chunks of 64
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `GradCache` | Main gradient cache implementation |
| `RandContext` | RNG state management for deterministic recomputation |
| `CacheFunc` | Wrapper for model forward/backward calls |

## Common Configurations

| Model | Batch | Chunk | Epochs | VRAM | Speed |
|-------|-------|-------|--------|------|-------|
| CLIP ViT-B | 128 | 64 | 3 | 20GB | 90ms |
| Qwen3-VL-2B | 128 | 64 | 10 | 25GB | 115ms |
| DINOv3-ViT-B | 256 | 64 | 20 | 40GB | 180ms |

## Advanced: Combining Optimizations

### Maximum Memory Efficiency
```yaml
# 3-layer optimization
use_gradient_cache: true
gradient_cache_chunk_size: 64
gradient_checkpointing: true  # +20% VRAM savings
use_lora: true                 # +10% VRAM savings
```

Total savings: **40-50% VRAM**

## Troubleshooting

**Q: Training is too slow**
A: Increase `gradient_cache_chunk_size` or reduce epochs.

**Q: Still out of memory**
A: Reduce `batch_size`, enable `gradient_checkpointing`, or use smaller model.

**Q: Different results with/without gradient cache**
A: This shouldn't happen. Check RNG seeding and floating-point precision.

## Related

- **Gradient Checkpointing**: Also reduces VRAM but slower
- **LoRA**: Further reduces trainable parameters
- **Mixed Precision**: Use bfloat16 for additional savings

---

::: vembed.grad_cache.grad_cache.GradCache
::: vembed.grad_cache.context_managers.RandContext
::: vembed.grad_cache.functional
