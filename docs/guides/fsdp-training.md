# FSDP Training Guide

Fully Sharded Data Parallel (FSDP) enables training of large models (8B+ parameters) across multiple GPUs by sharding model parameters, gradients, and optimizer states.

## When to Use FSDP

**Use FSDP when:**
- Training models with 8B+ parameters
- You have 4+ GPUs available
- GPU memory per GPU is 40GB+
- You want to scale horizontally without performance penalty

**Don't use FSDP when:**
- Model fits on a single GPU with DDP
- You need maximum training speed (FSDP has communication overhead)
- Debugging model issues (DDP is simpler)

## Quick Start: Train Qwen3-8B with FSDP

```bash
cd vembed-factory

# Distributed training with FSDP (8 GPUs)
accelerate launch vembed/entrypoints/train.py examples/qwen3_8b_fsdp.yaml

# Or using the helper script
bash examples/run_qwen3_8b_fsdp.sh
```

## Configuration

### FSDP Config File

Edit `configs/accelerate_fsdp.yaml`:

```yaml
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP    # Wrap transformer layers
  fsdp_sharding_strategy: FULL_SHARD               # Shard everything
  fsdp_cpu_ram_efficient_loading: true             # Use CPU for inactive params
  fsdp_offload_params: true                        # Offload to CPU if needed
  fsdp_backward_prefetch: BACKWARD_PRE             # Prefetch gradients
  fsdp_limit_all_gathers: true                     # Limit concurrent ops
  fsdp_min_num_params: 1000000                     # Min size for sharding
  fsdp_use_orig_params: true                       # Use original param objects

mixed_precision: bf16                               # Use bfloat16 for efficiency
num_processes: 8                                    # Number of GPUs (auto-detected)
```

### Training Config

Edit your training YAML (e.g., `examples/qwen3_8b_fsdp.yaml`):

```yaml
# Model
model_name: "Qwen/Qwen3-VL-Embedding-8B"
torch_dtype: "bfloat16"

# Training - IMPORTANT: tune these for your GPUs
batch_size: 16              # Per GPU, reduce if OOM
epochs: 3
learning_rate: 2.0e-5

# Memory Optimization
use_fsdp: true              # Enable FSDP
gradient_checkpointing: true
use_lora: true              # LoRA reduces trainable params

# Features
use_mrl: true
mrl_dims: [1024]           # Reduce from default for memory
```

## Memory Optimization Tips

### 1. Reduce Batch Size

If you get CUDA OOM errors:

```yaml
batch_size: 8   # Start small, increase if no OOM
```

Each GPU will get its own batch shard, so total BS = local_batch_size * num_gpus.

### 2. Enable Gradient Checkpointing

```yaml
gradient_checkpointing: true
```

Trades compute for memory by recomputing activations during backprop.

### 3. Reduce MRL Projection Dimension

```yaml
use_mrl: true
mrl_dims: [1024]  # Default 3584, reduce to 1024 for memory
```

MRL projection layers can consume significant memory.

### 4. Use LoRA

```yaml
use_lora: true
```

LoRA dramatically reduces trainable parameters and memory usage.

### 5. CPU Parameter Offloading

```yaml
# In configs/accelerate_fsdp.yaml
fsdp_offload_params: true
fsdp_cpu_ram_efficient_loading: true
```

Offloads inactive parameters to CPU. Trades some compute for memory.

## Common Issues and Solutions

### CUDA Out of Memory

**Symptom**: `torch.cuda.OutOfMemoryError`

**Solutions (in order):**
1. Reduce batch size: `batch_size: 8`
2. Enable gradient checkpointing: already on by default
3. Reduce MRL dims: `mrl_dims: [512]` or `[1024]`
4. Disable gradient cache (if enabled): `use_gradient_cache: false`

### NCCL Error: unhandled cuda error

**Symptom**: `RuntimeError: NCCL Error 1: unhandled cuda error`

**Solutions:**
1. Ensure all parameters have uniform dtype (automatic in our code)
2. Reduce batch size significantly: `batch_size: 8`
3. Check GPU connectivity: `nvidia-smi`

### Mixed Dtype Error

**Symptom**: `ValueError: Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32`

**Solution**: Our code automatically fixes this with `unify_model_dtype_for_fsdp()`.

## Performance Tips

### 1. Set Compute Device to GPU

```yaml
# Only in configs/accelerate_fsdp.yaml
compute_device: gpu
```

### 2. Use Flash Attention

```yaml
attn_implementation: "flash_attention_2"
```

Reduces memory and speeds up training.

### 3. Increase Learning Rate Slightly

FSDP has different gradient dynamics, try:

```yaml
learning_rate: 3.0e-5  # Slightly higher than DDP
```

### 4. Use Adam Optimizer

```yaml
optim: "adamw_8bit"    # 8-bit Adam for memory efficiency
```

## Monitoring FSDP Training

### Check GPU Memory Usage

```bash
# Terminal 1: Start training
accelerate launch vembed/entrypoints/train.py examples/qwen3_8b_fsdp.yaml

# Terminal 2: Monitor
watch -n 1 'nvidia-smi | grep -E "process|MiB"'
```

Expected: Each GPU uses 20-40GB (depending on batch size and model).

### Enable NCCL Debugging

If you encounter communication issues:

```bash
NCCL_DEBUG=TRACE accelerate launch vembed/entrypoints/train.py examples/qwen3_8b_fsdp.yaml 2>&1 | tee fsdp_trace.log
```

### View FSDP Diagnostics

Logs include:

```
FSDP dtype check: all parameters already in torch.bfloat16
FSDP parameter summary:
  Total: 8,192,000,000 (8.19B)
  Trainable: 8,192,000,000 (8.19B)
```

## FSDP vs Alternatives

| Method | Memory | Speed | Setup | Use When |
|--------|--------|-------|-------|----------|
| **FSDP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 8B+ models, high-end GPUs |
| **DDP + Gradient Cache** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 2B models, good GPU connectivity |
| **DDP** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Small models, simple setup |
| **Single GPU** | ⭐⭐ | ⭐⭐ | ⭐ | Debugging, small batches |

## Best Practices

1. **Start with small batch size**: FSDP works best with smaller batches (8-32 per GPU)
2. **Use bfloat16**: Reduces memory and speeds up training
3. **Enable gradient checkpointing**: Usually worth the compute/memory trade-off
4. **Monitor first training**: Watch GPU memory in first epoch
5. **Use NCCL backend**: Ensure `NCCL_DEBUG=INFO` for diagnostics
6. **Save checkpoints regularly**: `save_steps: 100`

## Example: Complete FSDP Training Setup

```bash
# 1. Prepare data
mkdir -p data
# Put your data in data/train.jsonl, data/val.jsonl

# 2. Create config
cat > my_fsdp_config.yaml << 'EOF'
model_name: "Qwen/Qwen3-VL-Embedding-8B"
data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
output_dir: "output"
batch_size: 16
epochs: 3
use_fsdp: true
gradient_checkpointing: true
use_lora: true
learning_rate: 2.0e-5
EOF

# 3. Train
accelerate launch vembed/entrypoints/train.py my_fsdp_config.yaml

# 4. Monitor output
tail -f output/training_log.txt
```

## Troubleshooting Checklist

- [ ] All GPUs visible: `nvidia-smi` shows all devices
- [ ] Correct accelerate config: `accelerate config` (should show FSDP)
- [ ] Batch size reduced to 16 or less
- [ ] Gradient checkpointing enabled
- [ ] Memory usage reasonable in first 10 steps
- [ ] No dtype mismatches in logs
- [ ] No NCCL errors in first epoch

See also:
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Hugging Face Accelerate FSDP Guide](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
- [Getting Started Guide](./getting-started.md)
