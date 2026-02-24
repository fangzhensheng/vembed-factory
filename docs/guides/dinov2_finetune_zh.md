# 微调 DINOv2：构建高精度的“以图搜图”引擎

**以图搜图 (Image Retrieval)** 是计算机视觉领域的一项核心技术，广泛应用于电商同款搜索、视觉搜索引擎等场景。其核心原理是将图像转换为向量 (Embedding)，通过计算向量相似度来召回相关图片。

Meta 推出的 **DINOv2** 是当前领先的自监督视觉大模型，生成的特征具有强大的语义表达能力。然而，通用的预训练模型在特定垂直领域（如细粒度商品检索）往往表现欠佳。为了区分相似但不同的商品（例如不同款式的椅子），我们需要对模型进行**微调 (Fine-tuning)**。

本文将介绍如何使用开源工具 **[vembed-factory](https://github.com/fangzhensheng/vembed-factory)**，在 Stanford Online Products (SOP) 数据集上微调 DINOv2 模型，显著提升其检索精度。

---

## 1. 为什么需要微调？

尽管 DINOv2 拥有强大的泛化能力，但它是基于通用海量数据预训练的。在面对特定领域的细粒度分类任务时（例如区分两个外观极其相似但型号不同的商品），直接使用预训练特征（Zero-shot）往往无法达到工业级的精度要求。

微调的核心目标是通过**对比学习 (Contrastive Learning)**，优化向量空间：
- **拉近**同一商品不同视角的图片向量距离（正样本对）。
- **推远**不同商品的图片向量距离（负样本对）。

实验表明，经过微调，模型在 SOP 数据集上的 Recall@1（首位命中率）可以从约 50% 提升至 80% 以上。

## 2. 环境准备

我们将使用 `vembed-factory` 框架，它封装了数据加载、Loss 计算、多卡训练等底层细节，使用户能专注于模型策略。

推荐使用 `uv` 进行环境管理，以获得更快的依赖安装体验：

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆项目
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# 3. 初始化环境
uv sync
source .venv/bin/activate
```

## 3. 数据集准备：SOP

本次实战使用 **Stanford Online Products (SOP)** 数据集，包含 120k 张图片，覆盖 22k 个商品类别。

### 3.1 下载数据
请从 [SOP 官方网站](https://cvgl.stanford.edu/projects/lifted_struct/) 或 Kaggle 下载数据集，并解压至 `data/stanford_online_products/` 目录：

```
data/stanford_online_products/
├── Ebay_train.txt      # 训练集索引
├── Ebay_test.txt       # 测试集索引
├── bicycle_final/      # 图片文件夹
├── chair_final/
...
```

### 3.2 数据预处理
原始数据为 txt 索引格式，我们需要将其转换为框架支持的 JSONL 格式。项目提供了预处理脚本：

```bash
# 运行数据预处理脚本
python examples/prepare_sop_data.py --sop_root data/stanford_online_products
```

运行完成后，目录下会生成 `train.jsonl` 和 `val.jsonl`，内容格式如下：

```json
{"query_image": "bicycle_final/111.jpg", "positive": "bicycle_final/112.jpg", "label": 1001}
```

该格式定义了检索任务中的正样本对（Query Image 与 Positive Image 属于同一商品）。

## 4. 模型微调

我们通过配置文件来定义训练参数，无需修改底层代码。

### 4.1 配置文件解析 (`examples/dinov2_i2i.yaml`)

```yaml
# --- 模型配置 ---
model_type: "custom"
model_name: "facebook/dinov2-small"  # 可根据显存选择 small/base/large
retrieval_mode: "i2i"                # Image-to-Image 检索模式
use_lora: true                       # 启用 LoRA 高效微调，降低显存占用

# --- 数据路径 ---
data_path: "data/stanford_online_products/train.jsonl"
image_root: "data/stanford_online_products"

# --- 训练参数 ---
epochs: 20
batch_size: 128                      # 较大的 batch size 有助于对比学习
loss_type: "infonce"                 # 使用 InfoNCE Loss
learning_rate: 5.0e-5
```

### 4.2 启动训练

确认配置无误后，执行以下命令启动训练：

```bash
bash examples/run_dinov2_i2i.sh
```

或者使用 CLI 命令：

```bash
uv run vembed train --config examples/dinov2_i2i.yaml
```

训练过程中，模型权重和日志将保存在 `output_sop_dinov2_i2i` 目录下。

## 5. 效果评测

训练完成后，我们需要在 SOP 测试集上评估模型的检索性能。

使用内置的评测脚本：

```bash
# --model_path 指向微调后的模型 checkpoint 路径
python benchmark/run.py sop \
    --model_path output_sop_dinov2_i2i/checkpoint-1000 \
    --sop_root data/stanford_online_products \
    --batch_size 128
```

### 预期结果

根据我们在 **DINOv2-base** 上的实验结果，微调带来的性能提升非常显著：

| 模型 | 指标 | Zero-shot (未微调) | Fine-tuned (微调后) | 提升幅度 |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv2-base** | **Recall@1** | 55.01% | **84.49%** | **+29.48%** |
| | Recall@10 | 71.09% | 94.00% | +22.91% |

即使使用参数量较小的 **DINOv2-small**，Recall@1 通常也能达到 **75%** 以上。这表明微调后的模型能更准确地识别细粒度的商品特征。

## 6. 总结

通过 **vembed-factory**，我们大大简化了视觉 Embedding 模型的微调流程：
1. 使用 `prepare_sop_data.py` 标准化数据格式。
2. 通过 `dinov2_i2i.yaml` 灵活配置模型参数。
3. 一键启动训练与评测。

这使得构建工业级的高精度以图搜图系统变得更加高效和便捷。

如果您对本项目感兴趣，欢迎访问 GitHub 仓库并点亮 Star：[vembed-factory](https://github.com/fangzhensheng/vembed-factory)
