# LoRA Fine-tuning Guide

Low-Rank Adaptation (LoRA) enables efficient fine-tuning by training only small rank adapters while keeping the base model frozen. This reduces memory usage and training time by 50-70%.

## When to Use LoRA

**Use LoRA when:**
- You want to fine-tune a large pre-trained model efficiently
- GPU memory is limited
- You need fast training iterations
- You want to store multiple LoRA adapters for different tasks

**Don't use LoRA when:**
- The base model architecture needs to change
- You need to modify layer normalization or embeddings
- You're training from scratch

## Quick Start

Enable LoRA in your config:

```yaml
use_lora: true

# Optional: customize LoRA parameters
lora_r: 16                           # Rank (higher = more params, more memory)
lora_alpha: 32                       # Scaling factor
lora_dropout: 0.05                   # Dropout for regularization
lora_target_modules:                 # Modules to apply LoRA to
  - "q_proj"
  - "v_proj"
  - "query"
  - "value"
  - "key"
  - "dense"
```

## Configuration

### YAML Configuration

Complete LoRA configuration:

```yaml
# Model
model_name: "Qwen/Qwen3-VL-Embedding-2B"
torch_dtype: "bfloat16"

# LoRA Configuration
use_lora: true
lora_r: 16                          # Rank of LoRA adapter
lora_alpha: 32                       # Scales LoRA by alpha/r
lora_dropout: 0.05                   # Dropout within adapter
lora_target_modules:                 # Which modules to apply LoRA
  - "q_proj"                         # Query projection
  - "v_proj"                         # Value projection
  - "query"                          # Alternative naming
  - "value"
  - "key"
  - "dense"

# Training
batch_size: 64
learning_rate: 1e-4                  # LoRA typically uses higher LR
epochs: 3

# Data
data_path: "data/train.jsonl"
output_dir: "output_lora"
```

### CLI Override

```bash
python run.py config.yaml \
  --config_override use_lora=true lora_r=32 lora_alpha=64
```

## LoRA Architecture

### How LoRA Works

```
Original layer:
  y = W * x  (W is frozen)

LoRA layer:
  y = W * x + (A @ B) * x  (A, B are trainable)

Where:
  W: original weight matrix (frozen, 8B parameters)
  A: low-rank down-projection (8B → rank-r)
  B: low-rank up-projection (rank-r → 8B)
```

### Rank Selection

| Rank | Memory | Accuracy | Speed | Use Case |
|------|--------|----------|-------|----------|
| **4** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Quick experiments |
| **8** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Standard fine-tuning |
| **16** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Quality important |
| **32** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | High-quality adaptation |
| **64** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Near full fine-tuning |

**Recommendation**: Start with rank=16, adjust based on:
- Available GPU memory
- Target accuracy
- Training time constraints

## Memory and Speed Benefits

### Example: Qwen3-VL-2B with LoRA

```
Without LoRA:
  Model parameters: 2B
  Optimizer states (Adam): 2B * 2 = 4B  (momentum, variance)
  Gradients: 2B
  Total memory: ~8B parameters need storage

With LoRA (rank=16):
  Model parameters: 2B (frozen, read-only)
  LoRA parameters: ~100M  (2B * 16 / 2B)
  Optimizer states: 100M * 2 = 200M
  Gradients: 100M
  Total memory: ~2.3B + 2B read-only = significant savings
```

**Expected speedup**: 1.5-2.0x faster training
**Expected memory**: 40-50% reduction

## Target Modules

LoRA can be applied to specific modules to optimize efficiency:

### Vision-Language Models (Qwen, LLaVA)
```yaml
lora_target_modules:
  - "q_proj"      # Query projection
  - "v_proj"      # Value projection
  - "query"       # Alternative naming
  - "value"
  - "key"
```

### CLIP Models
```yaml
lora_target_modules:
  - "q_proj"
  - "v_proj"
  - "out_proj"
```

### Text Models (BERT, Qwen)
```yaml
lora_target_modules:
  - "query"
  - "key"
  - "value"
  - "dense"
```

### Keep Some Modules Untrainable

```yaml
lora_target_modules:
  - "q_proj"
  - "v_proj"
  # Exclude: encoder, embeddings, layer_norm
modules_to_save:
  - "classifier"     # Save these in LoRA adapter
  - "pooler"
```

## Hyperparameter Tuning

### Learning Rate

LoRA typically uses higher learning rates than full fine-tuning:

```yaml
# Full fine-tuning
learning_rate: 1e-5

# LoRA fine-tuning
learning_rate: 1e-4 to 1e-3
```

Start with 1e-4 and adjust based on validation loss.

### Rank and Alpha

```yaml
lora_r: 16
lora_alpha: 32           # Often lora_alpha = 2 * lora_r
```

The effective scaling is: `(lora_alpha / lora_r)`

If `alpha = 2*r`, scaling = 2x (constant scaling regardless of rank)

### Dropout

```yaml
lora_dropout: 0.05       # 5% dropout
```

Prevents overfitting. Increase to 0.1 for small datasets.

## Example Configurations

### Quick Experiment (Memory Constrained)

```yaml
model_name: "Qwen/Qwen3-VL-Embedding-2B"
use_lora: true
lora_r: 8
lora_alpha: 16
batch_size: 32
learning_rate: 1e-4
epochs: 1
```

### Production Fine-tuning (Quality Priority)

```yaml
model_name: "Qwen/Qwen3-VL-Embedding-2B"
use_lora: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.1
batch_size: 64
learning_rate: 5e-5
epochs: 3
gradient_checkpointing: true
```

### Multi-GPU with FSDP

```yaml
model_name: "Qwen/Qwen3-VL-Embedding-8B"
use_fsdp: true
use_lora: true
lora_r: 16
lora_alpha: 32
batch_size: 16
learning_rate: 2e-5
epochs: 3
gradient_checkpointing: true
```

## Training LoRA Models

### Train

```bash
# Simple training
python run.py config_lora.yaml

# Distributed with DDP
accelerate launch vembed/entrypoints/train.py config_lora.yaml

# Distributed with FSDP + LoRA
accelerate launch vembed/entrypoints/train.py examples/qwen3_8b_fsdp.yaml
```

### Monitor Training

```bash
tail -f output/training_log.txt
```

### Save Checkpoints

LoRA adapters are saved alongside the base model:

```
output/
├── checkpoint-1000/
│   ├── adapter_config.json    # LoRA configuration
│   ├── adapter_model.bin      # LoRA weights only (~50MB)
│   └── training_args.bin
```

## Using LoRA Models

### Load and Inference

```python
from transformers import AutoModel
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "output/checkpoint-3000")

# Inference
model.eval()
with torch.no_grad():
    output = model(input_ids)
```

### Merge LoRA into Base Model

```python
# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("output/merged_model")
```

### Use with vembed-factory

```python
from vembed import Predictor

# Works seamlessly with LoRA checkpoints
predictor = Predictor(model_path="output/checkpoint-3000")

text_emb = predictor.encode_text("a cat")
image_emb = predictor.encode_image("cat.jpg")
```

## Best Practices

1. **Start with rank=16**: Good balance of efficiency and quality
2. **Use high learning rate**: 1e-4 to 1e-3 range
3. **Enable gradient checkpointing**: Combine with LoRA for max efficiency
4. **Use bfloat16**: Always train in bfloat16 with LoRA
5. **Validate early**: Check validation loss after epoch 1
6. **Save adapters separately**: Don't merge unless necessary for deployment

## Combining LoRA with Other Techniques

### LoRA + Gradient Cache

For large batch training with limited memory:

```yaml
use_lora: true
use_gradient_cache: true    # Works well together
batch_size: 32
```

### LoRA + Gradient Checkpointing

For deep models:

```yaml
use_lora: true
gradient_checkpointing: true
batch_size: 64
```

### LoRA + MRL

For dimension-flexible embeddings:

```yaml
use_lora: true
use_mrl: true
mrl_dims: [1024, 768, 512, 256]
```

### LoRA + FSDP + Gradient Cache (Maximum Efficiency)

For 8B models:

```yaml
use_fsdp: true
use_lora: true
use_gradient_cache: false    # Disabled automatically with FSDP
gradient_checkpointing: true
batch_size: 16
```

## Troubleshooting

### Training Too Slow

**Problem**: LoRA training not faster than expected

**Solutions:**
- Increase batch size
- Reduce rank (from 32 to 16)
- Use gradient checkpointing

### Validation Loss Not Improving

**Problem**: Model not learning effectively

**Solutions:**
- Increase learning rate (try 5e-4)
- Increase rank (from 8 to 16 or 32)
- Train longer (more epochs)
- Check data quality

### CUDA OOM with LoRA

**Problem**: "CUDA out of memory" even with LoRA

**Solutions:**
- Reduce batch size
- Reduce rank
- Enable gradient checkpointing
- Use gradient cache

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Getting Started Guide](./getting-started.md)
- [FSDP Training Guide](./fsdp-training.md)
