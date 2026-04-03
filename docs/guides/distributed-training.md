# Distributed Training Guide

A comprehensive guide to choosing and configuring distributed training strategies in vembed-factory.

## Quick Decision Tree

```text
Do you have multiple GPUs?
├─ No  → Single GPU training
│       └─ Can still use DDP on 1 GPU (no benefit, but works)
│
└─ Yes → How many?
   ├─ 2-4 GPUs, model fits on 1 GPU?
   │  └─ Use DDP (simple, fast)
   │
   ├─ 4-8 GPUs, model 2-8B?
   │  └─ Use DDP + Gradient Cache (balance of memory and effective batch size)
   │
   └─ 8+ GPUs, model > 8B (e.g., Qwen3-VL-72B)?
      └─ Use FSDP + Gradient Cache (The Ultimate Combo for huge models)
```

## Training Strategies Comparison

### 1. Single GPU (Baseline)

**When**: Debugging, small models, testing

```bash
vembed train config.yaml
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
```text
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
# Modern CLI interface (recommended)
accelerate launch vembed/entrypoints/train.py config.yaml \
    --use_gradient_cache \
    --gradient_accumulation_steps 4

# Or legacy format
accelerate launch vembed/entrypoints/train.py config.yaml \
    --config_override use_gradient_cache=true gradient_accumulation_steps=4
```

**How it works:**
Gradient Cache decouples the forward pass from the backward pass by caching activations. This allows you to simulate a massive batch size (e.g., 512+) for contrastive learning while only keeping a small chunk (e.g., 8) in memory for the backward pass.

---

### 4. FSDP + Gradient Cache (The Ultimate Combo)

**When**: You are training massive multimodal models (8B - 72B) and need both model sharding and huge effective batch sizes for InfoNCE loss.

**Architecture:**
```text
- FSDP shards the model weights, gradients, and optimizer states across all GPUs.
- Gradient Cache chunks the input data to keep activation memory low.
- vembed-factory safely orchestrates the `no_sync` contexts and `accelerator.accumulate` to prevent state machine crashes.
```

**Setup:**
Just enable both in your YAML config:

```yaml
use_fsdp: true
use_gradient_cache: true
batch_size: 1  # Physical batch size per chunk
gradient_accumulation_steps: 128  # Will result in Effective Batch Size = 1 * 128 * Num_GPUs
```

**Note on Compatibility:**
In older versions, FSDP and Gradient Cache were mutually exclusive. We have completely refactored the underlying mixed-precision and synchronization engine to make them work flawlessly together.

---

## Memory Comparison

### Example: Training Qwen3-2B on different setups

| Setup | Num GPU | Per-GPU Memory | Total Batch | Speed | Notes |
|-------|---------|----------------|-------------|-------|-------|
| Single GPU | 1 | 40GB | 1 | 1x | Baseline |
| DDP | 4 | 12GB | 256 (64*4) | 3.8x | Each GPU has full model |
| DDP + GradCache | 4 | 10GB | 512 (64*4*2) | 3.5x | 2x effective batch |
| FSDP | 4 | 5GB | 256 (64*4) | 3x | Model sharded |
| FSDP + GradCache | 4 | 4GB | 1024 (64*4*4) | 2.5x | Best balance |

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

**FSDP + Gradient Cache:**
```
Per-GPU memory = (model_params / num_gpu) * 2 + (batch_size/chunks) * param_per_sample + (optimizer_state / num_gpu)
               = (2B/4)*2 + (64/8)*small + (2B*2)/4 = ~4GB per GPU
               (But process up to 16x larger effective batch)
```

---

## Choosing the Right Strategy

### Decision Matrix

| Model Size | GPU Count | Recommended | Alternative |
|------------|-----------|------------|-------------|
| < 500M | 1-2 | Single GPU or DDP | N/A |
| 500M-2B | 2-4 | DDP | N/A |
| 2B-8B | 4-8 | DDP + GradCache | FSDP (if memory tight) |
| 8B-20B | 8-16 | FSDP + GradCache + LoRA | DDP + GradCache (slower) |
| 20B+ | 16+ | FSDP + GradCache + LoRA | Only option |

### Real-World Recommendations

**GPU Allocation:**
- 2x A100 40GB: DDP with CLIP, batch_size=256
- 4x A100 80GB: DDP + GradCache with Qwen3-2B, effective_batch=512
- 8x A100 80GB: FSDP + GradCache with Qwen3-8B, effective_batch=1024
- 16x H100 80GB: FSDP + GradCache + LoRA for Qwen3-72B

**Budget Optimization:**
- Memory constraint → Use FSDP if model > 50% available GPU memory
- Speed constraint → Use DDP (no communication overhead)
- Balanced → Use DDP + GradCache (best of both)
- Huge models → Use FSDP + GradCache (only option)

---

## Practical Examples

### Example 1: Fast Training (DDP with CLIP)

```bash
# Setup: 4x A100 40GB
cat > clip_ddp.yaml << 'EOF'
model_name: "openai/clip-vit-base-patch32"
batch_size: 128
learning_rate: 1e-4
epochs: 3
loss_type: "infonce"
EOF

accelerate config  # Select: multi-GPU → DDP
accelerate launch vembed/entrypoints/train.py clip_ddp.yaml
```

**Expected**: ~2-3 hours on 4 GPUs, throughput ~1000 samples/sec

---

### Example 2: Balanced (DDP + GradCache with Qwen3-2B)

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

accelerate config  # Select: multi-GPU → DDP
accelerate launch vembed/entrypoints/train.py qwen3_balanced.yaml
```

**Expected**: ~6 hours on 4 GPUs, effective batch=512, throughput ~100 samples/sec

---

### Example 3: Large Model (FSDP + GradCache with Qwen3-8B)

```bash
# Setup: 8x A100 80GB
cat > qwen3_8b_fsdp.yaml << 'EOF'
model_name: "Qwen/Qwen3-VL-Embedding-8B"
batch_size: 1
use_fsdp: true
use_gradient_cache: true
gradient_accumulation_steps: 128
use_lora: true
lora_r: 16
gradient_checkpointing: true
learning_rate: 2e-5
epochs: 3
EOF

accelerate config  # Select: multi-GPU → FSDP → transformer_based_wrap
accelerate launch vembed/entrypoints/train.py qwen3_8b_fsdp.yaml
```

**Expected**: ~10 hours on 8 GPUs, effective batch=1024, throughput ~80 samples/sec

---

## Performance Tuning

### Maximize Speed

1. **Increase batch size** (until OOM)
2. **Reduce gradient checkpointing** (trades memory for speed)
3. **Use smaller precision** (bfloat16 is default)
4. **Use smaller models** (CLIP is faster than Qwen3)

### Maximize Memory Efficiency

1. **Use FSDP** for models > available GPU memory
2. **Enable gradient cache** for large effective batches
3. **Use LoRA** instead of full fine-tuning
4. **Enable gradient checkpointing**
5. **Reduce batch size** (trades speed for memory)

### Best Memory + Speed Balance

1. **Use DDP + GradCache** for 2-8B models (recommended)
2. **Tune batch size** to 32-64 per GPU
3. **Enable gradient checkpointing**
4. **Set gradient_cache_chunk_size** to balance OOM vs recomputation

---

## Troubleshooting

### Slow Training

**Check:**
1. GPU utilization: Should be > 80% (use `nvidia-smi` or `torch.cuda.utilization()`)
2. Communication overhead: FSDP slower than DDP (expected ~10-20% slowdown)
3. I/O bottleneck: Data loading speed (check with `data_loading_steps`)

**Solutions:**
1. Increase batch size (you have memory budget)
2. Switch to DDP if using FSDP (faster but higher memory)
3. Use faster storage (SSD vs HDD)
4. Enable mixed precision (bfloat16)

### Unstable Training (Loss NaN)

**Check:**
1. Learning rate too high (typical: 1e-5 to 2e-5)
2. Gradient explosion (check with `max_grad_norm`)
3. Data issues (corrupted images, missing values)

**Solutions:**
1. Reduce learning rate (divide by 2-4)
2. Enable gradient clipping: `max_grad_norm: 1.0`
3. Check data quality with `vembed validate-data`

### CUDA OOM

**Check:**
1. Model size vs available GPU memory
2. Batch size too large
3. Gradient accumulation too aggressive

**Solutions:**
1. Reduce `batch_size` (per-GPU)
2. Reduce `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Enable `use_gradient_cache: true`
5. Switch to FSDP if model > 50% GPU memory

See [FSDP Training Guide](./fsdp-training.md) for more FSDP-specific troubleshooting.

---

## References

- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [Gradient Cache Paper](https://arxiv.org/abs/2205.11342)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)
