# Examples Configuration Guide

Complete guide for choosing and using vembed-factory configurations.

## Quick Navigation

### New Users? Start Here
👉 **[Quick Start Guide](quickstart/README.md)** - 3-step setup in 5 minutes

### Configuration Management (NEW)
- **`dataset_info.json`** - Dataset registry (replaces hardcoded paths in shell scripts)
- **`training_info.json`** - Training configuration templates
- **`config_manager.py`** - Python utility to manage configurations

**Use**: `python config_manager.py` to list available datasets and training configs

### Decision Tree: Which Configuration to Use?

```
Are you new to vembed-factory?
├─ YES → Start with quickstart/clip_minimal.yaml
└─ NO → Continue...

Do you have a specific model in mind?
├─ CLIP → models/clip/
├─ Qwen3-VL → models/qwen3_vl/
├─ DINOv2 → models/dinov2/
├─ SigLIP → models/siglip/
└─ Other → models/other/

Do you need multi-GPU training?
├─ YES → distributed/
└─ NO → models/[model]/base.yaml

Do you want to use a special training strategy?
├─ Hard negatives → strategies/hard_negative/
├─ Knowledge distillation → strategies/knowledge_distillation/
├─ Special loss functions → strategies/special_loss/
├─ Late interaction (ColBERT) → strategies/late_interaction/
└─ None → Use base config from models/
```

## Directory Structure

### `quickstart/` - Get Started Quickly
Minimal configurations designed for fast testing and validation.

- **`clip_minimal.yaml`** - CLIP with minimal setup (recommended for beginners)
- **`qwen3_vl_minimal.yaml`** - Qwen3-VL lightweight training
- **`README.md`** - Quick start guide with 3 steps

**Use when**: Testing, learning, or validating your setup

### `models/` - Model-Specific Configurations

#### `clip/`
- **`base.yaml`** - Standard CLIP training
- **`with_wandb.yaml`** - CLIP with W&B experiment tracking
- **`coco.yaml`** - COCO dataset specialized config

#### `qwen3_vl/`
- **`2b_base.yaml`** - Lightweight 2B model (DDP/single-GPU)
- **`8b_base.yaml`** - Full 8B model (DDP on 4+ GPUs)
- **`8b_fsdp.yaml`** - 8B with FSDP for extreme scale

#### `dinov2/`
- **`i2i.yaml`** - DINOv2 image-to-image retrieval
- **`v3_i2i.yaml`** - DINOv3 improved version
- **`bert.yaml`** - DINOv2 + BERT composed encoder
- **`colbert.yaml`** - DINOv2 with ColBERT late-interaction

#### `siglip/`
- **`base.yaml`** - SigLIP with sigmoid loss

#### `other/`
- **`mae_i2i.yaml`** - MAE for image retrieval
- **`bert_t2t.yaml`** - BERT for text-to-text
- **`bge_t2t.yaml`** - BGE for semantic search

**Use when**: You've chosen a specific model and want standard training

### `strategies/` - Advanced Training Techniques

#### `hard_negative/`
Mine difficult samples for better model convergence.

- **`infonce.yaml`** - InfoNCE with hard negative mining
- **`qwen3_vl.yaml`** - Memory bank based hard mining

#### `in_batch_hard/`
Select hard negatives from current batch (no extra memory).

- **`qwen3_vl.yaml`** - In-batch hard negative mining

#### `knowledge_distillation/`
Transfer knowledge from large models to small ones.

- **`clip.yaml`** - CLIP student trained by ViT-L teacher

#### `special_loss/`
Experiment with specialized loss functions.

- **`clip_cosent.yaml`** - CoSENT loss for contrastive learning
- **`clip_triplet.yaml`** - Triplet loss with hard mining

#### `late_interaction/`
Token-level rather than sentence-level interactions.

- **`qwen.yaml`** - Qwen2-7B with ColBERT
- **`dinov2.yaml`** - DINOv2 with ColBERT

**Use when**: You want to experiment with training strategies beyond basic setup

### `distributed/` - Multi-GPU Training

- **`fsdp.yaml`** - FSDP for sharding across many GPUs
- **`with_tracking.yaml`** - Distributed training with experiment tracking (W&B/SwanLab)

**Use when**: Training large models across multiple GPUs or need experiment tracking

## Configuration Selection Flowchart

### By Use Case

**👶 Learning & Testing**
```yaml
→ quickstart/clip_minimal.yaml
```

**🎯 Production Training (Single GPU)**
```yaml
Select model: models/[model]/base.yaml
Optional: Add strategy from strategies/
```

**🚀 Large-Scale Training (4+ GPUs)**
```yaml
Base: models/[model]/[large_variant].yaml
Strategy: distributed/fsdp.yaml
```

**📊 Experiment Tracking**
```yaml
Base: models/[model]/base.yaml
Tracking: distributed/with_tracking.yaml (override report_to)
```

## Common Configuration Patterns

### Pattern 1: Basic Single-GPU Training
```bash
python train.py --config_path examples/models/clip/base.yaml
```

### Pattern 2: Multi-GPU with DDP (2-8 GPUs)
```bash
torchrun --nproc_per_node=4 train.py \
  --config_path examples/models/qwen3_vl/8b_base.yaml
```

### Pattern 3: Large Model with FSDP (8+ GPUs)
```bash
torchrun --nproc_per_node=8 train.py \
  --config_path examples/distributed/fsdp.yaml
```

### Pattern 4: With Experiment Tracking
```bash
python train.py --config_path examples/models/clip/with_wandb.yaml \
  --wandb_project my_project
```

## Quick Reference Table

| Task | Config | GPUs | Time |
|------|--------|------|------|
| Test setup | `quickstart/clip_minimal.yaml` | 1x24GB | ~5 min |
| Quick experiment | `models/clip/base.yaml` | 1x40GB | ~30 min |
| Production CLIP | `models/clip/base.yaml` | 1x40GB+ | Variable |
| Production Qwen3-VL 2B | `models/qwen3_vl/2b_base.yaml` | 1x40GB+ | Variable |
| Production Qwen3-VL 8B | `models/qwen3_vl/8b_base.yaml` | 4x40GB+ | Variable |
| Large-scale training | `distributed/fsdp.yaml` | 8x80GB+ | Variable |

## Tips for Success

### Memory Management
- **24GB GPU**: Use gradient cache (`use_gradient_cache: true`)
- **40GB GPU**: Can disable gradient cache for faster training
- **80GB+ GPU**: Use without any memory optimization

### Training Speed
- Increase `num_workers` for faster data loading
- Use DDP/FSDP for multi-GPU training
- Enable mixed precision (`torch_dtype: bfloat16`)

### Model Selection
- **CLIP**: Fastest, best for image-text pairs
- **SigLIP**: Stable, similar to CLIP with sigmoid loss
- **Qwen3-VL**: Multimodal, slower but more capable
- **DINOv2**: Image-only, fast and accurate for vision tasks

### Loss Functions
- **InfoNCE**: Default, works well for most cases
- **Hard negatives**: Improve ranking, require tuning
- **ColBERT**: Late-interaction, better for dense retrieval
- **Triplet**: For metric learning, needs careful margin tuning

## Modifying Configurations

Each YAML file can be overridden from command line:

```bash
python train.py \
  --config_path examples/models/clip/base.yaml \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --epochs 10
```

## For More Details

- **Model-specific tuning**: See `models/[model]/README.md`
- **Strategy details**: See `strategies/[strategy]/README.md`
- **Distributed setup**: See `distributed/README.md`
- **Full options**: Check main documentation

## Troubleshooting

**Q: Which config should I start with?**
A: Always start with `quickstart/clip_minimal.yaml` to test your setup.

**Q: How do I choose between models?**
A: CLIP is fastest. Qwen3-VL is more capable but slower. DINOv2 for image-only tasks.

**Q: Can I combine strategies?**
A: Yes! Load a base config and override with strategy settings.

**Q: What's the difference between base.yaml and minimal.yaml?**
A: Minimal uses smaller batch sizes and fewer epochs for quick testing.

---

**Happy training! 🚀**

Need help? Check the main README.md or open an issue on GitHub.
