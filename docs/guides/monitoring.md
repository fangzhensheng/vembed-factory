# Experiment Monitoring Guide

vembed-factory provides seamless integration with popular experiment tracking platforms: **Weights & Biases (WandB)**, **SwanLab**, and **TensorBoard**.

## Choosing a Tracker

You can configure the tracker using the `report_to` parameter in your YAML config or CLI:

```yaml
report_to: "wandb"    # Options: wandb, swanlab, tensorboard, all, none
```

## Configuring Experiment Metadata

To keep your experiments organized, you can pass metadata directly in your training configuration. These will be automatically propagated to your chosen dashboard.

```yaml
# In your config.yaml
report_to: "swanlab"
run_name: "qwen3-vl-2b-coco-lora-v1"
run_tags: ["qwen3", "lora", "coco", "contrastive"]
run_notes: "First attempt at fine-tuning Qwen3-VL on COCO with rank=16"
```

## Supported Platforms

### 1. SwanLab (Recommended for domestic users)
SwanLab is an excellent lightweight alternative, especially for users in regions with restricted access to WandB.

**Setup:**
```bash
pip install swanlab
swanlab login
```

**Config:**
```yaml
report_to: "swanlab"
```

### 2. Weights & Biases (WandB)
The industry standard for experiment tracking.

**Setup:**
```bash
pip install wandb
wandb login
```

**Config:**
```yaml
report_to: "wandb"
```

### 3. TensorBoard
For local offline monitoring.

**Setup:**
```bash
pip install tensorboard
```

**Config:**
```yaml
report_to: "tensorboard"
```

To view logs:
```bash
tensorboard --logdir ./output_dir/runs/
```

## Tracked Metrics

By default, the framework logs:
- `train/loss`: The contrastive loss value
- `train/learning_rate`: Current learning rate (useful for checking warmup/cosine schedules)
- `train/epoch`: Current epoch
- `train/global_step`: Total optimization steps
- `eval/loss`: Validation loss (if validation data is provided)
- `eval/recall@k`: Validation metrics (if evaluator is configured)
