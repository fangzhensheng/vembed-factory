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
- clip/ - CLIP系列
- qwen3_vl/ - Qwen-VL系列
- dinov2/ - DINOv2系列
- siglip/ - SigLIP
- other/ - 其他模型

### strategies/ - 按策略分类
- hard_negative/ - 硬负样本挖掘
- in_batch_hard/ - 批内硬负样本
- knowledge_distillation/ - 知识蒸馏
- special_loss/ - 特殊Loss函数
- late_interaction/ - 晚期交互

### distributed/ - 分布式训练
FSDP和多GPU配置

## 🎯 如何选择配置？

**我有16GB显存** → `models/clip/base.yaml`
**我有24GB显存** → `models/qwen3_vl/2b_base.yaml`
**我有多块GPU** → `distributed/fsdp.yaml`
**我是新手** → `quickstart/clip_minimal.yaml`
**我想要硬负样本** → `strategies/hard_negative/`
**我想要知识蒸馏** → `strategies/knowledge_distillation/`

## 📖 使用方法

```bash
# 基础用法
python run.py examples/models/clip/base.yaml

# 命令行覆盖参数
python run.py examples/models/clip/base.yaml \
    --learning_rate=5e-5 \
    --batch_size=64 \
    --num_epochs=10

# 恢复中断的训练
python run.py examples/models/clip/base.yaml \
    --resume_from_checkpoint outputs/model/checkpoint-1000
```

## 📚 原始配置文件

原始的27个YAML文件都保留在examples/目录根目录中，可以继续使用。
新的组织结构是为了更好地帮助用户快速找到合适的配置。

---

提示：从quickstart/开始，然后根据需要调整参数。祝训练顺利！ 🚀
