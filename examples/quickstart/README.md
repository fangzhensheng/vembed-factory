# 快速开始（5分钟）

欢迎使用vembed-factory！本指南将帮助你在5分钟内完成第一次训练。

## 📋 前置条件

- Python 3.10+
- PyTorch with CUDA
- 至少16GB显存（推荐24GB+）

## 🚀 5分钟快速开始

### Step 1: 准备数据（1分钟）

创建 `my_data.jsonl` 文件，每行一个JSON对象：

```jsonl
{"query": "a photo of a dog", "positive": "path/to/dog.jpg"}
{"query": "a cat sitting on a chair", "positive": "path/to/cat.jpg"}
{"query": "sunset over mountains", "positive": "path/to/sunset.jpg"}
```

或者下载示例数据：
```bash
cd examples
python prepare_data.py  # 下载COCO或Flickr30k
```

### Step 2: 选择配置（1分钟）

**选项A：CLIP（推荐，最快）**
```bash
cp examples/quickstart/clip_minimal.yaml my_config.yaml
```

**选项B：Qwen-VL（更准确，需要24GB显存）**
```bash
# 从models/qwen3_vl/2b_base.yaml复制
cp examples/models/qwen3_vl/2b_base.yaml my_config.yaml
```

### Step 3: 修改配置（1分钟）

编辑 `my_config.yaml`，修改数据路径：

```yaml
data_path: "path/to/my_data.jsonl"  # ← 改成你的数据路径
output_dir: "output/my_first_training"
```

其他常见参数：
```yaml
batch_size: 32          # 根据显存调整（16GB用32，24GB用64）
num_epochs: 3           # 训练轮数
learning_rate: 5e-5     # 学习率
```

### Step 4: 启动训练（2分钟）

```bash
python run.py my_config.yaml
```

你将看到：
```
✓ Configuration validated
✓ Model loaded: openai/clip-vit-base-patch32
✓ Data loaded: 1000 samples
✓ Training started
[████████████────] 60% | Step 300/500 | Loss: 0.25 | ETA: 10m
```

## ✅ 完成！

训练完后，你可以在 `output/my_first_training/` 找到：
- `model_best/` - 最佳模型权重
- `training_logs.txt` - 训练日志
- `config.yaml` - 使用的配置

## 📚 下一步

- **调整参数**：修改 `batch_size`、`learning_rate` 等参数
- **用不同模型**：查看 `examples/models/` 下的其他配置
- **用特殊策略**：查看 `examples/strategies/` 下的硬负样本、知识蒸馏等
- **分布式训练**：查看 `examples/distributed/` 使用多GPU

## 🆘 常见问题

**Q: 显存不够？**
A: 减小 `batch_size`（例如从32改为16），或启用 `use_gradient_cache: true`

**Q: 训练太慢？**
A: 启用 `use_gradient_cache: true` 或用CLIP模型而不是Qwen-VL

**Q: 数据格式不对？**
A: 查看 `examples/prepare_data.py` 或 README中的数据格式说明

**Q: 怎么恢复中断的训练？**
A: 添加参数 `--resume_from_checkpoint output/my_first_training/checkpoint-500`

---

祝你训练顺利！有问题可以查看 `examples/README.md` 或项目README。🚀
