# vembed-factory 配置示例指南

欢迎！这里包含了vembed-factory的所有配置示例。本指南将帮助你快速找到合适的配置。

## 🚀 快速开始（推荐）

如果你是第一次使用vembed-factory，直接从这里开始：

```bash
# 查看快速开始配置
cat examples/quickstart/README.md
```

## 📁 目录结构

### quickstart/ - 快速开始
最小化的配置示例，适合新手快速启动训练。

### models/ - 按模型分类
- clip/ - CLIP系列 (vision_clip_*.yaml)
- qwen3_vl/ - Qwen-VL系列 (multimodal_qwen3_vl_*.yaml)
- dinov2/ - DINOv2系列 (vision_dinov2_*.yaml)
- siglip/ - SigLIP (vision_siglip_*.yaml)
- other/ - 其他模型 (text_bert_*, vision_mae_*)

### strategies/ - 按策略分类
- hard_negative/ - 硬负样本挖掘 (strategy_hardneg_*.yaml)
- in_batch_hard/ - 批内硬负样本 (strategy_inbatch_*.yaml)
- knowledge_distillation/ - 知识蒸馏 (strategy_distill_*.yaml)
- special_loss/ - 特殊Loss函数 (strategy_loss_*.yaml)
- late_interaction/ - 晚期交互 (strategy_lateint_*.yaml)

### datasets/ - 按数据集分类
- dataset_coco_t2i.yaml - COCO数据集（图文）
- dataset_flickr30k_t2i.yaml - Flickr30k数据集（图文）
- dataset_msmarco_t2t.yaml - MS MARCO数据集（纯文本）

### distributed/ - 分布式训练
FSDP和多GPU配置

## 🎯 如何选择配置？

**我有16GB显存** → `models/clip/vision_clip_base.yaml`
**我有24GB显存** → `models/qwen3_vl/multimodal_qwen3_vl_2b_base.yaml`
**我有多块GPU** → `models/qwen3_vl/multimodal_qwen3_vl_8b_fsdp.yaml`
**我是新手** → `quickstart/clip_minimal.yaml`
**我想要硬负样本** → `strategies/hard_negative/strategy_hardneg_*.yaml`
**我想要知识蒸馏** → `strategies/knowledge_distillation/strategy_distill_*.yaml`

## 📖 使用方法

**查看可用配置和数据集：**

```bash
# 列出所有训练配置
vembed list-configs

# 列出所有数据集
vembed list-datasets

# 查看某个数据集详情
vembed show-dataset flickr30k_t2i
```

**训练模型：**

```bash
# 基础用法
vembed train examples/models/clip/vision_clip_base.yaml

# 命令行覆盖参数
vembed train examples/models/clip/vision_clip_base.yaml \
    --learning-rate 5e-5 \
    --batch-size 64 \
    --epochs 10

# 恢复中断的训练
vembed train examples/models/clip/vision_clip_base.yaml \
    --resume-from-checkpoint outputs/model/checkpoint-1000

# 干运行（只生成配置，不训练）
vembed train examples/models/clip/vision_clip_base.yaml --dry-run
```

## 📚 原始配置文件

原始的27个YAML文件都保留在examples/目录根目录中，可以继续使用。
新的组织结构是为了更好地帮助用户快速找到合适的配置。

---

提示：从quickstart/开始，然后根据需要调整参数。祝训练顺利！ 🚀
