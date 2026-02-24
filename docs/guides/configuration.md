# Configuration Guide

vembed-factory uses a hierarchical configuration system with three dataclasses:
- `ModelArguments` - Model selection and architecture
- `DataArguments` - Data sources and paths
- `TrainingArguments` - Training hyperparameters

## Quick Start

### Via YAML Config

Create `config.yaml`:

```yaml
model_name_or_path: "openai/clip-vit-base-patch32"
data_path: "data/train.jsonl"
output_dir: "output"
epochs: 3
batch_size: 32
learning_rate: 1e-5
retrieval_mode: "t2i"
loss_type: "infonce"
scheduler_type: "cosine"
warmup_ratio: 0.1
```

Train:
```bash
python run.py config.yaml
```

### Via CLI

```bash
python run.py \
  --model_name_or_path "openai/clip-vit-base-patch32" \
  --data_path "data/train.jsonl" \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 1e-5
```

### Via Python API

```python
from vembed import Trainer

trainer = Trainer("openai/clip-vit-base-patch32")
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    epochs=3,
    batch_size=32,
    learning_rate=1e-5,
    retrieval_mode="t2i",
    loss_type="infonce"
)
```

## Configuration Hierarchy

`hparams.py` defines three main argument classes:

### ModelArguments

```yaml
model_name_or_path: "openai/clip-vit-base-patch32"  # HuggingFace model or local path
encoder_mode: "auto"  # auto|clip_like|composed|vlm_generic
use_lora: false  # Parameter-efficient fine-tuning
teacher_model_name: null  # For knowledge distillation
projection_dim: null  # Optional: project embeddings to this dimension
```

**Encoder Modes:**
- `"auto"`: Auto-detect from model type (CLIP, Qwen, etc.)
- `"clip_like"`: CLIP-style dual encoders
- `"composed"`: Mix text encoder + image encoder
- `"vlm_generic"`: Generic vision-language models
- `"qwen3_vl"`: Optimized for Qwen3-VL-Embedding

### DataArguments

```yaml
data_path: "data/train.jsonl"
val_data_path: null  # Optional validation set
image_root: "./"  # Root directory for relative image paths
```

### TrainingArguments

Core hyperparameters:

```yaml
epochs: 3
batch_size: 32
learning_rate: 1e-5
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
```

**Retrieval Mode** (determines data format):

```yaml
retrieval_mode: "t2i"  # t2i|i2i|i2t|t2t|m2i|m2t
```

**Loss Function**:

```yaml
loss_type: "infonce"  # infonce|mrl|triplet|cosent|colbert|distillation
temperature: 0.07
```

**Scheduler**:

```yaml
scheduler_type: "cosine"  # cosine|linear|constant|constant_with_warmup
warmup_ratio: 0.1  # Fraction of total steps for warmup
```

**Memory Optimization**:

```yaml
use_gradient_cache: false  # Enable for large effective batch sizes
use_mrl: false  # Matryoshka Representation Learning
gradient_checkpointing: false  # Reduce memory at computational cost
```

**Advanced Options**:

```yaml
torch_dtype: "float32"  # float32|bfloat16
attn_implementation: "sdpa"  # sdpa|flash_attention_2
```

## Configuration Presets

Preset YAML files apply model-specific defaults:

- `configs/defaults.yaml` - Base configuration for all models
- `configs/clip.yaml` - OpenAI CLIP optimized settings
- `configs/siglip.yaml` - Google SigLIP settings
- `configs/qwen3.yaml` - Qwen3-VL-Embedding settings

**Example preset (configs/qwen3.yaml):**

```yaml
model_name_or_path: "Qwen/Qwen3-VL-Embedding-2B"
batch_size: 16
learning_rate: 1e-5
use_gradient_cache: true
```

## LoRA Fine-tuning

Enable parameter-efficient fine-tuning:

```yaml
use_lora: true
lora_r: 8  # Low-rank dimension
lora_alpha: 16  # Scaling factor
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj", "o_proj"]
```

## Matryoshka Representation Learning (MRL)

Train a model at multiple embedding dimensions:

```yaml
use_mrl: true
mrl_dims: [768, 512, 256]  # Dimensions to train
```

At inference, extract embeddings at any configured dimension:

```python
predictor = Predictor("output")
emb_768 = predictor.encode_text("query", dimension=768)
emb_256 = predictor.encode_text("query", dimension=256)
```

## Knowledge Distillation

Train a smaller student from a larger teacher:

```yaml
loss_type: "distillation"
teacher_model_name: "Qwen/Qwen3-VL-Embedding-2B"
distillation_alpha: 0.7  # Weight: 0.7*distillation + 0.3*task_loss
distillation_temperature: 4.0  # Softness of teacher targets
```

## Experiment Tracking

### Weights & Biases

```yaml
report_to: "wandb"
run_name: "clip-finetune-v1"
project: "embedding-training"
```

Then login and train:
```bash
wandb login
python run.py config.yaml
```

### TensorBoard

```yaml
report_to: "tensorboard"
logging_dir: "logs/"
```

View with:
```bash
tensorboard --logdir logs/
```

## Full Configuration Reference

See `vembed/hparams.py` for all available parameters:

```python
@dataclass
class ModelArguments:
    model_name_or_path: str
    encoder_mode: str = "auto"
    use_lora: bool = False
    # ... more parameters

@dataclass
class DataArguments:
    data_path: str
    val_data_path: Optional[str] = None
    image_root: str = "./"
    # ... more parameters

@dataclass
class TrainingArguments:
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-5
    # ... more parameters
```

## Configuration Merging Order

1. Load defaults from `configs/defaults.yaml`
2. Apply preset if model matches (e.g., `configs/clip.yaml`)
3. Merge user YAML config file
4. Apply CLI overrides and `--config_override` arguments
5. Final config written to `.train_config.yaml` in output directory

**Example:**

```bash
# Merges: defaults < clip preset < config.yaml < CLI args
python run.py config.yaml --batch_size 64
```

The CLI argument `--batch_size 64` overrides any value in config.yaml.

## Environment Variables

Set defaults via environment:

```bash
export VEMBED_BATCH_SIZE=16
export VEMBED_LEARNING_RATE=1e-5
export VEMBED_EPOCHS=5

python run.py config.yaml  # Uses env values as defaults
```

## Saving Configuration

The merged configuration is automatically saved:

```bash
# After training
cat output/.train_config.yaml
```

Use this to reproduce training:

```bash
python run.py output/.train_config.yaml
```

## Tips

- Start with a preset YAML and modify specific values
- Use `--config_override param1=value1 param2=value2` for quick experiments
- Enable W&B (`report_to: wandb`) to track experiments
- Use gradient caching (`use_gradient_cache: true`) for large effective batch sizes
