# 微调 DINOv3：构建高精度的”以图搜图”引擎

**以图搜图 (Image Retrieval)** 是计算机视觉领域的一项核心技术，广泛应用于电商同款搜索、视觉搜索引擎等场景。其核心原理是将图像转换为向量 (Embedding)，通过计算向量相似度来召回相关图片。

Meta 推出的 **DINOv3** 是当前最新一代的自监督视觉大模型，相比前一代 DINOv2 具有更强的语义表达和检索能力。然而，通用的预训练模型在特定垂直领域（如细粒度商品检索）往往表现欠佳。为了区分相似但不同的商品（例如不同款式的椅子），我们需要对模型进行**微调 (Fine-tuning)**。

本文将介绍如何使用开源工具 **[vembed-factory](https://github.com/fangzhensheng/vembed-factory)**，在 Stanford Online Products (SOP) 数据集上微调 DINOv3 模型，显著提升其检索精度。

---

## 1. 为什么需要微调？

尽管 DINOv3 拥有强大的泛化能力，但它是基于通用海量数据预训练的。在面对特定领域的细粒度分类任务时（例如区分两个外观极其相似但型号不同的商品），直接使用预训练特征（Zero-shot）往往无法达到工业级的精度要求。

微调的核心目标是通过**对比学习 (Contrastive Learning)**，优化向量空间：
- **拉近**同一商品不同视角的图片向量距离（正样本对）。
- **推远**不同商品的图片向量距离（负样本对）。

实验表明，经过微调，模型在 SOP 数据集上的 Recall@1（首位命中率）可以从约 65% 提升至 83% 以上。

## 2. 环境准备

我们将使用 `vembed-factory` 框架，它封装了数据加载、Loss 计算、多卡训练等底层细节，使用户能专注于模型策略。

推荐使用 `uv` 进行环境管理，以获得更快的依赖安装体验：

```bash
# 1. 克隆项目
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# 2. 初始化环境（使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# 3. 或者使用传统 pip 环境
pip install -e .
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

原始数据为 txt 索引格式，我们需要将其转换为框架支持的 JSONL 格式。项目提供了统一的数据预处理脚本：

```bash
# 运行数据预处理脚本，自动下载或处理已有数据
python examples/prepare_data.py sop_i2i
```

> **提示**：若数据尚未下载，脚本会自动通过 `kagglehub` 从 Kaggle 下载 SOP 数据集。

运行完成后，目录下会生成 `train.jsonl` 和 `val.jsonl`，内容格式如下：

```json
{
  "query_image": "bicycle_final/111.jpg",
  "positive": "bicycle_final/112.jpg",
  "label": 1001
}
```

**数据格式说明：**
- `query_image` - 查询图片路径（相对路径）
- `positive` - 正样本图片路径（同一商品的另一视角）
- `label` - 商品 ID（用于 Supervised Contrastive Loss）

训练时，框架会自动从同一批次的其他样本中生成**负样本**（in-batch negatives），无需预先指定。

## 4. 模型微调

我们通过配置文件来定义训练参数，无需修改底层代码。框架支持灵活的参数覆盖机制，所有参数都可通过 CLI 动态调整。

### 4.0 参数配置说明

主要参数含义如下：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_name` | 必需 | 预训练模型名称（HuggingFace 格式） |
| `retrieval_mode` | t2i | 检索模式：`i2i`（图搜图）、`t2i`（文搜图）等 |
| `data_path` | data/train.jsonl | 训练集 JSONL 文件路径 |
| `val_data_path` | null | 验证集路径（可选） |
| `image_root` | "" | 图片基础路径 |
| `output_dir` | output | 输出目录（保存 checkpoint 和日志） |
| `epochs` | 3 | 训练轮次 |
| `batch_size` | 32 | 批大小 |
| `learning_rate` / `lr` | 2.0e-5 | 学习率 |
| `loss_type` | infonce | 损失函数：`infonce`、`triplet`、`cosent`、`colbert` 等 |
| `use_lora` | false | 是否启用 LoRA 参数高效微调 |
| `save_steps` | 0 | 每 N 步保存一次 checkpoint（0 = 仅保存 epoch 末） |
| `logging_steps` | 10 | 每 N 步记录一次日志 |
| `report_to` | none | 实验跟踪：`wandb`、`tensorboard`、`none` 等 |

### 4.1 配置文件解析 (`examples/dinov3_i2i.yaml`)

```yaml
# --- 模型配置 ---
model_name: facebook/dinov3-vitb16-pretrain-lvd1689m  # DINOv3 ViT-B/16（最新最强）
retrieval_mode: i2i                                    # Image-to-Image 检索模式
use_lora: true                                         # 启用 LoRA 高效微调，降低显存占用

# --- 数据路径 ---
data_path: data/stanford_online_products/train.jsonl
val_data_path: data/stanford_online_products/val.jsonl
image_root: data/stanford_online_products
output_dir: experiments/output_sop_dinov3_i2i

# --- 训练参数 ---
epochs: 20
batch_size: 128                                        # 较大的 batch size 有助于对比学习
loss_type: infonce                                     # 使用 InfoNCE Loss + Supervised Contrastive
learning_rate: 0.0001                                 # 学习率
logging_steps: 10                                      # 每 10 步记录一次日志
save_steps: 500                                        # 每 500 步保存一次 checkpoint
report_to: none                                        # 不使用 wandb/tensorboard 等实验跟踪
```

### 4.2 启动训练

确认配置无误后，执行以下命令启动训练：

```bash
bash examples/run_dinov3_i2i.sh
```

或者直接使用 Python 运行脚本：

```bash
python run.py examples/dinov3_i2i.yaml
```

如果需要覆盖配置文件中的参数，可以使用 CLI 参数：

```bash
python run.py examples/dinov3_i2i.yaml --config_override epochs=30 batch_size=64
```

训练过程中，模型权重和日志将保存在 `experiments/output_sop_dinov3_i2i` 目录下。

## 5. 效果评测

训练完成后，我们需要在 SOP 测试集上评估模型的检索性能。

使用内置的评测脚本：

```bash
# --model_path 指向微调后的模型 checkpoint 路径
python benchmark/run.py sop \
    --model_path experiments/output_sop_dinov3_i2i/checkpoint-1000 \
    --sop_root data/stanford_online_products \
    --batch_size 128
```

或者通过配置覆盖进行评测：

```bash
python run.py examples/dinov3_i2i.yaml --config_override lr=0 model_name_or_path=experiments/output_sop_dinov3_i2i/checkpoint-1000
```

### 预期结果

根据我们在 **Stanford Online Products (SOP)** 数据集上的实验结果，微调带来的性能提升非常显著：

| 模型 | 指标 | Zero-shot (未微调) | Fine-tuned (微调后) | 提升幅度 |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv3-ViT-B/16** | **Recall@1** | 65.32% | **83.13%** | **+17.81%** |
| *(facebook/dinov3-vitb16-pretrain-lvd1689m)* | Recall@10 | 80.73% | **93.34%** | **+12.61%** |
| | Recall@100 | 90.43% | **97.26%** | **+6.83%** |

**训练配置：**
- 模型：facebook/dinov3-vitb16-pretrain-lvd1689m（最新）
- 数据集：SOP 训练集（~120k 张图片）
- 训练轮次：20 epochs
- Batch Size：128
- LoRA：启用

微调后的 DINOv3-ViT-B/16 在 Recall@1 上相比零样本预训练模型获得了 **17.81%** 的显著提升，达到 **83.13%** 的高精度，可广泛应用于生产环境中的商品检索系统。

## 6. 常见问题

### 6.1 如何调整训练参数？

方式一：编辑 YAML 配置文件
```bash
# 修改 examples/dinov3_i2i.yaml 中的参数，然后运行
python run.py examples/dinov3_i2i.yaml
```

方式二：使用 CLI 参数覆盖
```bash
# 无需修改配置文件，直接通过 --config_override 传递参数
python run.py examples/dinov3_i2i.yaml \
    --config_override epochs=30 batch_size=64 learning_rate=0.00005
```

### 6.2 如何使用其他模型？

修改 `model_name` 参数：

```bash
# 使用 DINOv2-base（上一代模型）
python run.py examples/dinov3_i2i.yaml --config_override model_name=facebook/dinov2-base

# 使用 DINOv3-ViT-S/14（更小、更快）
python run.py examples/dinov3_i2i.yaml --config_override model_name=facebook/dinov3-vits14-pretrain-lvd1689m
```

**可选模型及性能对比：**

| 模型 | 参数量 | 显存占用 | Recall@1 | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- |
| DINOv2-base | ~86M | 中等 | 79.97% | 上一代 |
| **DINOv3-ViT-B/16**（默认） | ~87M | 中等 | **83.13%** | **最新推荐** ✓ |
| DINOv3-ViT-S/14 | ~21M | 低 | ~81% | 边缘设备、快速迭代 |

根据我们的实验结果，**DINOv3-ViT-B/16** 在 SOP 数据集上微调后能达到 83.13% 的 Recall@1，是生产环境的最佳选择。

### 6.3 如何在多卡上训练？

框架使用 `accelerate` 自动检测并配置分布式训练。

**方式一：直接使用训练脚本（推荐）**
```bash
# 框架会自动检测可用 GPU，启用分布式训练
bash examples/run_dinov3_i2i.sh
```

**方式二：使用 accelerate 手动配置**
```bash
# 交互式配置分布式训练参数
accelerate config

# 启动训练（会自动使用多卡）
accelerate launch run.py examples/dinov3_i2i.yaml
```

> **提示**：大多数情况下，直接运行脚本即可自动利用所有 GPU。如遇到特殊硬件配置或分布式策略需求，才需要手动 `accelerate config`。

### 6.4 输出目录结构是什么？

训练后会生成：
```
experiments/output_sop_dinov3_i2i/
├── checkpoint-1000/        # 保存的 checkpoint
│   ├── config.json
│   ├── adapter_config.json  # LoRA 配置（如果使用）
│   └── adapter_model.bin    # LoRA 权重（如果使用）
├── checkpoint-2000/
├── train_config.yaml        # 训练时的完整配置
└── train_results.json       # 训练指标
```

### 6.5 数据集是如何处理的？

**数据流程：**
```
原始 SOP 数据 (Ebay_train.txt / Ebay_test.txt)
    ↓
prepare_data.py sop_i2i 转换
    ↓
JSONL 格式 (query_image, positive, label)
    ↓
GenericRetrievalDataset 加载
    ↓
训练循环提取 label（支持 Supervised Contrastive Loss）
    ↓
Supervised Contrastive Loss：
  - label 相同的样本 → 视为正样本对（拉近距离）
  - label 不同的样本 → 视为负样本对（推远距离）
    ↓
✓ 模型学习更强的类别级别判别特征
```

**关键点：**
- 每个样本在 JSONL 中只有 **1 个显式正样本**（`positive` 字段，同商品另一视角）
- **隐式正样本**：批次内所有 label 相同的其他样本
- **负样本**：批次内所有 label 不同的样本
- **label 的作用**：通过 **Supervised Contrastive Loss**，将同类别的所有样本拉近，不同类别样本推远，大幅提升类别判别能力

## 7. 总结

通过 **vembed-factory**，我们大大简化了视觉 Embedding 模型的微调流程：

1. **数据准备** - `python examples/prepare_data.py sop_i2i`
   - 自动下载或处理 SOP 数据集
   - 转换为简洁的 JSONL 格式（query_image, positive, label）

2. **配置管理** - YAML 配置文件 + CLI 参数覆盖
   - 灵活指定模型、损失函数、学习率等
   - 无需修改代码即可调整训练策略

3. **模型训练** - 一键启动
   - 支持单卡 / 多卡分布式训练
   - 支持 LoRA 参数高效微调
   - 自动 checkpoint 保存

4. **效果评测** - 内置评测脚本
   - 在 SOP 测试集上评估 Recall@1、Recall@10 等指标
   - 对标 Zero-shot 预训练模型，通常获得 20-30% 的提升

**核心优势：**
- ✅ 简洁的数据格式（无需复杂的 hard negatives 配置）
- ✅ 完整的 Supervised Contrastive Learning 支持
- ✅ In-batch negatives 自动生成
- ✅ 支持多种损失函数和检索模式

这使得构建工业级的高精度以图搜图系统变得更加高效和便捷。

如果您对本项目感兴趣，欢迎访问 GitHub 仓库并点亮 Star：[vembed-factory](https://github.com/fangzhensheng/vembed-factory)
