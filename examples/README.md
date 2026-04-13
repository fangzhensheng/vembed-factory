# vembed-factory 配置示例指南

docker push ai-harbor.facethink.com/mlops-public/prepare-agent-04-v3

## 目录结构

### `quickstart/`

docker -harbor.com/mlops-public/prepare-agent-v2:04-09-v3

- `clip_minimal.yaml`
- `qwen3_vl_minimal.yaml`
- `debug_minimal.yaml`

### `strategies/`

docker push ai-harbor.facethink.com/mlops-public/prepare-agent-v2:04-09-v3

- `hard_negative/`
- `in_batch_hard/`
- `knowledge_distillation/`
- `late_interaction/`
- `special_loss/`

### `distributed/`

docker push ai-harbor.facethink.com/mlops-public/prepare-agent-v2:04-09-v3

- `qwen3_vl_8b_fsdp.yaml`
- `qwen3_vl_2b_wandb.yaml`
- `qwen3_vl_2b_swanlab.yaml`

### `datasets/`

docker push ai-harbor.facethink.com/mlops-public/prepare-agent-v2:04-09-v3

- `dataset_coco_t2i.yaml`
- `dataset_flickr30k_t2i.yaml`
- `dataset_msmarco_t2t.yaml`

## 典型选择

- **第一次使用**: `examples/quickstart/clip_minimal.yaml`
- **想试多模态 VLM**: `examples/quickstart/qwen3_vl_minimal.yaml`
- **需要 FSDP**: `examples/distributed/qwen3_vl_8b_fsdp.yaml`
- **需要知识蒸馏**: `examples/strategies/knowledge_distillation/strategy_distill_clip_distillation.yaml`
- **需要 ColBERT/late interaction**: `examples/strategies/late_interaction/strategy_lateint_dinov2_colbert.yaml`

## 使用方式

```bash
# 基础训练
vembed train examples/quickstart/clip_minimal.yaml

# 覆盖参数
vembed train examples/quickstart/clip_minimal.yaml     --learning_rate 5e-5     --batch_size 64     --epochs 10

# 干运行
vembed train examples/quickstart/clip_minimal.yaml --dry_run

# 查看可用配置
vembed list-configs
```

check `examples/quickstart/README.md`。
