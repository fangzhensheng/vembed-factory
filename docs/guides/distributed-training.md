# Distributed Training Guide

A comprehensive guide to choosing and configuring distributed training strategies in vembed-factory.

## Quick Decision Tree

```
Do you have multiple GPUs?
├─ No  → Single GPU training
│       └─ Can still use DDP on 1 GPU (no benefit, but works)
│
└─ Yes → How many?
   ├─ 2-4 GPUs, model fits on 1 GPU?
   │  └─ Use DDP (simple, fast)
   │
   ├─ 4-8 GPUs, model > 30GB?
   │  └─ Use DDP + Gradient Cache (best balance)
   │
   └─ 8+ GPUs, model > 50GB (8B+)?
      └─ Use FSDP (only option for huge models)
```

## Training Strategies Comparison

### 1. Single GPU (Baseline)

**When**: Debugging, small models, testing

```bash
python run.py config.yaml
```

**Pros:**
- Simplest setup
- Easy debugging
- No communication overhead

**Cons:**
- Limited by single GPU memory
- Slow training

**Memory limit**: ~80GB per GPU max

---

### 2. DDP (Distributed Data Parallel)

**When**: 2-4 GPUs, model fits on one GPU

```bash
accelerate launch vembed/entrypoints/train.py config.yaml
```

**Architecture:**
```
GPU 0: Full model + batch 0 data
GPU 1: Full model + batch 1 data
GPU 2: Full model + batch 2 data
GPU N: Full model + batch N data
       ↓ sync gradients after each step
```

**Pros:**
- Simple setup
- Maximum training speed
- Minimal code changes

**Cons:**
- Each GPU stores full model (memory expensive)
- Not suitable for very large models

**Memory per GPU**: Full model + gradients + optimizer state
- Example: CLIP 100M model on 2 GPUs = ~2GB per GPU
- Example: Qwen3-8B on 8 GPUs = ~2GB per GPU + gradients = ~6GB

**Setup:**

```bash
# Create accelerate config
accelerate config  # Select "multi-GPU" → "DDP"

# Train
accelerate launch vembed/entrypoints/train.py config.yaml
```

---

### 3. DDP + Gradient Cache

**When**: 4-8 GPUs, model 2-8B, want large effective batch size

```bash
# Automatically enabled when gradient_checkpointing + model > 1B
python run.py config.yaml --config_override use_gradient_cache=true
```

**How it works:**
```
For batch_size=64, gradient_cache chunks=8:

Iteration 1: Process 8 samples, compute forward but NOT backward
Iteration 2: Process 8 samples, compute forward but NOT backward
...
Iteration 8: Process 8 samples, compute forward + backward
            ↓ Sync gradients from all chunks
End: Effective batch size = 64 samples
```

**Pros:**
- Much larger effective batch size without OOM
- Better generalization (larger batches help)
- Good speed (better than FSDP for 2-8B models)
- Works with existing DDP setup

**Cons:**
- Slower than vanilla DDP (recomputation overhead)
- Requires gradient cache compatible loss function
- Need to tune chunk size

**Configuration:**

```yaml
batch_size: 64                      # Per-GPU batch size
use_gradient_cache: true            # Enable gradient cache
gradient_cache_chunk_size: 8        # How many sub-steps

# Automatically compatible with gradient checkpointing
gradient_checkpointing: true
```

**Memory**: Lower than DDP alone
- Trades memory for computation
- Each chunk processed independently
- Total memory ≈ memory for chunk_size samples + model

---

### 4. FSDP (Fully Sharded Data Parallel)

**When**: 8+ GPUs, model 8B+, need maximum memory efficiency

```bash
accelerate config  # Select "multi-GPU" → "FSDP"
accelerate launch vembed/entrypoints/train.py config.yaml
```

**Architecture:**
```
GPU 0: Model shard 0 + optimizer state shard 0 + batch data 0
GPU 1: Model shard 1 + optimizer state shard 1 + batch data 1
...
GPU N: Model shard N + optimizer state shard N + batch data N
       ↓ all-gather for forward pass
       ↓ reduce-scatter for backward pass
```

**Pros:**
- Enables training huge models (100B+)
- Best memory efficiency
- Scales to many GPUs

**Cons:**
- More complex setup
- Communication overhead (slower than DDP)
- Requires careful configuration
- NCCL errors are harder to debug

**Memory per GPU**: ~model_size / num_gpus + gradients + optimizer
- Example: 8B model on 8 GPUs = 1GB model + 1GB gradients + 2GB optimizer = ~4GB

**Setup:**

```yaml
# In config
use_fsdp: true
gradient_checkpointing: true
use_lora: true               # Recommended with FSDP

# Automatically handled
batch_size: 16
```

---

## Detailed Strategy Guide

### Strategy 1: DDP (Simple)

**Best for:** 2-4 GPUs, models < 5B

**Configuration:**

```yaml
# config.yaml
model_name: "openai/clip-vit-base-patch32"
batch_size: 64
learning_rate: 1e-5
epochs: 3
use_gradient_cache: false   # No need
```

**Training:**

```bash
accelerate config
# Select: multi-GPU → DDP → No distributed training on CPU
accelerate launch vembed/entrypoints/train.py config.yaml
```

**Typical setup:**
- V100/A100 8x40GB: batch_size 64-128
- RTX 3090 8x24GB: batch_size 32-64

---

### Strategy 2: DDP + Gradient Cache (Recommended for 4-8B)

**Best for:** 4-8 GPUs, models 2-8B, want effective BS > 256

**Why**:
- Larger batches improve generalization
- Faster than FSDP
- Simpler than FSDP
- Works well for 2-8B models

**Configuration:**

```yaml
# config.yaml
model_name: "Qwen/Qwen3-VL-Embedding-2B"
batch_size: 64                      # Per-GPU batch size
use_gradient_cache: true            # Enable memory optimization
gradient_cache_chunk_size: 8        # Process 8x64=512 effective samples
gradient_checkpointing: true        # Double memory savings
learning_rate: 1e-5
epochs: 3
```

**Training:**

```bash
accelerate config
# Select: multi-GPU → DDP → Default distributed training
accelerate launch vembed/entrypoints/train.py config.yaml
```

**Memory calculation:**
- Per-GPU memory: ~model_size + gradients + (batch_size/chunk_size)*optimizer_state
- Example: 2B model, BS=64, chunks=8 → ~2GB + 2GB + 2GB = 6GB per GPU

---

### Strategy 3: FSDP (Necessary for 8B+)

**Best for:** 8+ GPUs, models 8B+

**Configuration:**

```yaml
# config.yaml
model_name: "Qwen/Qwen3-VL-Embedding-8B"
batch_size: 16                      # Smaller with FSDP
use_fsdp: true                      # Enable FSDP
gradient_checkpointing: true        # Memory optimization
use_lora: true                       # Recommended: reduces trainable params
learning_rate: 2e-5
epochs: 3
```

**Training:**

```bash
accelerate config
# Select: multi-GPU → FSDP → transformer_based_wrap
accelerate launch vembed/entrypoints/train.py config.yaml
```

**Tuning:**

```yaml
# If OOM:
batch_size: 8                       # Reduce batch size
lora_r: 8                          # Reduce LoRA rank
mrl_dims: [512]                    # Reduce MRL dimension

# If fast enough but OOM:
gradient_checkpointing: true        # Already on, but double-check
lora_r: 32                         # Increase for quality
```

---

## Memory Comparison

### Example: Training Qwen3-2B on different setups

| Setup | Num GPU | Per-GPU Memory | Total Batch | Speed | Notes |
|-------|---------|----------------|-------------|-------|-------|
| Single GPU | 1 | 40GB | 1 | 1x | Baseline |
| DDP | 4 | 12GB | 256 (64*4) | 3.8x | Each GPU has full model |
| DDP + GradCache | 4 | 10GB | 512 (64*4*2) | 3.5x | 2x effective batch |
| FSDP | 4 | 5GB | 256 (64*4) | 3x | Model sharded |

### Memory Formula

**DDP:**
```
Per-GPU memory = model_params * 2 + batch_size * param_per_sample + optimizer_state
               = 2B*2 + 64*small + 2B*2 = ~10GB per GPU
```

**DDP + Gradient Cache:**
```
Per-GPU memory = model_params * 2 + (batch_size/chunks) * param_per_sample + optimizer_state
               = 2B*2 + (64/8)*small + 2B*2 = ~10GB per GPU
               (But process 8x larger effective batch)
```

**FSDP:**
```
Per-GPU memory = (model_params / num_gpu) * 2 + batch_size * param_per_sample + (optimizer_state / num_gpu)
               = (2B/4)*2 + 64*small + (2B*2)/4 = ~5GB per GPU
```

---

## Choosing the Right Strategy

### Decision Matrix

| Model Size | GPU Count | Recommended | Alternative |
|------------|-----------|------------|-------------|
| < 500M | 1 | Single GPU | N/A |
| 500M-2B | 2-4 | DDP | N/A |
| 2B-8B | 4-8 | DDP + GradCache | FSDP (if memory tight) |
| 8B+ | 8-16 | FSDP + LoRA | DDP + GradCache (slower) |
| 50B+ | 16+ | FSDP + LoRA | Only option |

### Real-World Recommendations

**GPU Allocation:**
- 2x A100 40GB: DDP with CLIP, batch_size=256
- 4x A100 80GB: DDP + GradCache with Qwen3-2B, batch_size=256
- 8x A100 80GB: FSDP with Qwen3-8B, batch_size=16 → effective 128

**Budget Optimization:**
- Memory constraint: Use FSDP if model > available GPU memory
- Speed constraint: Use DDP (no communication overhead)
- Balanced: Use DDP + GradCache (best of both)

---

## Practical Examples

### Example 1: Fast Training (DDP)

```bash
# Setup: 4x A100 40GB
cat > clip_ddp.yaml << 'EOF'
model_name: "openai/clip-vit-base-patch32"
batch_size: 128
learning_rate: 1e-4
epochs: 3
loss_type: "infonce"
EOF

accelerate launch vembed/entrypoints/train.py clip_ddp.yaml
```

Expected: ~2-3 hours on 4 GPUs

### Example 2: Balanced (DDP + GradCache)

```bash
# Setup: 4x A100 80GB
cat > qwen3_balanced.yaml << 'EOF'
model_name: "Qwen/Qwen3-VL-Embedding-2B"
batch_size: 64
use_gradient_cache: true
gradient_cache_chunk_size: 8
gradient_checkpointing: true
learning_rate: 1e-5
epochs: 3
EOF

accelerate launch vembed/entrypoints/train.py qwen3_balanced.yaml
```

Expected: ~6 hours on 4 GPUs, effective batch=512

### Example 3: Large Model (FSDP)

```bash
# Setup: 8x A100 80GB
cat > qwen3_8b_fsdp.yaml << 'EOF'
model_name: "Qwen/Qwen3-VL-Embedding-8B"
batch_size: 16
use_fsdp: true
use_lora: true
lora_r: 16
gradient_checkpointing: true
learning_rate: 2e-5
epochs: 3
EOF

accelerate launch vembed/entrypoints/train.py qwen3_8b_fsdp.yaml
```

Expected: ~8 hours on 8 GPUs

---

## Performance Tuning

### Maximize Speed

1. **Increase batch size** (until OOM)
2. **Reduce gradient checkpointing** (trades memory for speed)
3. **Use flash_attention_2** if available
4. **Use smaller models** (CLIP vs Qwen)

### Maximize Memory Efficiency

1. **Use FSDP** for huge models
2. **Enable gradient cache** for large batches
3. **Use LoRA** instead of full fine-tuning
4. **Enable gradient checkpointing**
5. **Use smaller precision** (bfloat16)

### Best Memory + Speed Balance

1. **Use DDP + GradCache** for 2-8B models
2. **Tune batch size** to 32-64 per GPU
3. **Enable gradient checkpointing**
4. **Use bfloat16 precision**

---

## Troubleshooting

### Slow Training

**Check:**
1. GPU utilization: Should be > 80%
2. Communication overhead: FSDP slower than DDP
3. I/O bottleneck: Data loading speed

**Solutions:**
1. Increase batch size
2. Switch to DDP if using FSDP
3. Use faster storage (SSD vs HDD)

### Unstable Training (Loss NaN)

**Check:**
1. Learning rate too high
2. Gradient explosion
3. Data issues

**Solutions:**
1. Reduce learning rate (divide by 2-4)
2. Enable gradient clipping (max_grad_norm=1.0)
3. Check data quality

### CUDA OOM

See [FSDP Training Guide](./fsdp-training.md#common-issues-and-solutions)

---

## References

- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [Gradient Cache Paper](https://arxiv.org/abs/2205.11342)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)
