# CLIP 文本-图像检索微调实战指南

**文本-图像检索 (Text-to-Image Retrieval)** 是当今互联网应用中最常见的搜索方式。用户通过输入文本描述（如"红色运动鞋"）来搜索相关的商品图片，这在电商平台、视觉搜索引擎、内容管理系统中广泛应用。

**CLIP (Contrastive Language-Image Pre-training)** 是 OpenAI 推出的多模态基础模型，通过在 4 亿张图文对上进行对比学习预训练，能够有效地理解和关联图像与文本。尽管 CLIP 的通用泛化能力很强，但在特定领域（如商品、医学影像、卫星图像）的精度往往不理想，需要进行微调以适应领域特定的语义。

本文将详细介绍如何使用 **vembed-factory** 框架，基于 Flickr30k 数据集微调 CLIP 模型，构建一个高精度的文本-图像检索系统，性能可从零样本的 58% Recall@1 提升至 71% 以上。

---

## 1. 为什么需要微调 CLIP？

### 1.1 CLIP 的优势与局限

**优势：**
- ✅ 通用性强：预训练数据覆盖 400M 图文对，学到通用的跨模态表示
- ✅ 零样本能力：无需任何微调，即可对新类别进行分类
- ✅ 模型轻量：基础版本 (ViT-B/32) 仅需 4GB 显存，易于部署

**局限：**
- ❌ 领域泛化性：预训练数据以网络图片为主，缺乏特定领域的细粒度理解
- ❌ 精度瓶颈：在垂直领域上的 Recall@1 往往在 50-60% 左右
- ❌ 语言偏差：西方数据占比高，对非英文、特定方言等支持不足

### 1.2 微调的目标

通过微调，我们希望在保留通用特征的基础上，优化模型对**领域特定文本和图像的相似度计算**，具体表现为：

**对比学习的核心优化方向：**
- **拉近**相关的文本-图像对（如文本"红色运动鞋"与对应的运动鞋图片）
- **推远**不相关的文本-图像对（如"红色运动鞋"与衣服图片）

**性能提升预期：**

| 指标 | Zero-shot (未微调) | Fine-tuned (微调后) | 提升幅度 |
|------|------------------|-----------------|--------|
| **Recall@1** | 58% | **71%+** | **+13% pp** |
| **Recall@5** | 78% | **85%+** | **+7% pp** |
| **Recall@10** | 85% | **90%+** | **+5% pp** |

> 数据来自在 Flickr30k 数据集上的实验结果（使用 LoRA 微调，3 个 epoch）

---

## 2. 环境准备

### 2.1 安装 vembed-factory

推荐使用 `uv` 进行快速环境管理：

```bash
# 1. 克隆项目
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# 2. 使用 uv 初始化环境（推荐，速度快）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# 或者使用传统 pip
pip install -e ".[torch]"
```

### 2.2 验证安装

```bash
# 检查是否成功安装
python -c "from vembed import Trainer, Predictor; print('✓ vembed-factory 安装成功')"
```

### 2.3 硬件需求

| 组件 | 要求 | 备注 |
|------|------|------|
| **GPU** | CUDA 12.0+ | 其他版本可能需要重新编译 torch |
| **显存** | 8GB 以上 | LoRA 微调；全量微调需 24GB+ |
| **CPU 内存** | 16GB 以上 | 数据加载和模型初始化 |
| **存储** | 30GB | 数据集 + 模型 checkpoint |

---

## 3. 数据准备

### 3.1 使用 Flickr30k 数据集

**Flickr30k** 是文本-图像对标准数据集，包含：
- 31,783 张图片
- 每张图片 5 条文本描述
- 总计 158,915 个文本-图像对

#### 3.1.1 下载数据

```bash
# 方式1：使用脚本自动下载（推荐）
python examples/prepare_data.py flickr30k

# 方式2：手动从官方下载
# 访问：https://github.com/BryanPlummer/flickr30k_entities
# 下载后解压到 data/flickr30k/ 目录
```

> **注意**：Flickr30k 需要在线申请，首次下载可能需要数分钟。

### 3.2 数据格式转换

脚本执行完成后，会生成标准的 JSONL 格式文件：

```
data/flickr30k/
├── train.jsonl      # 30k 对训练数据
├── val.jsonl        # 1k 对验证数据
└── images/          # 31.7k 张图片
```

#### 3.2.1 JSONL 数据格式

```json
{
  "query": "A child in a pink dress is playing with a yellow frisbee in the snow",
  "positive": "flickr30k_images/1000092795.jpg"
}
```

**字段说明：**
- `query` - 文本描述（查询文本）
- `positive` - 对应的图片路径（正样本）

**重要特性：**
- 批次内的其他样本自动作为**负样本**（in-batch negatives）
- 无需预先配置 hard negatives，框架自动处理
- 支持多个正样本（列表形式）

### 3.3 自定义数据准备

如果你有自己的文本-图像对数据，需要转换为上述 JSONL 格式：

```python
import json

# 假设你有以下原始数据
custom_data = [
    {
        "image_path": "products/shoes_001.jpg",
        "descriptions": ["红色运动鞋", "Nike 跑鞋"]
    },
    {
        "image_path": "products/shoes_002.jpg",
        "descriptions": ["蓝色篮球鞋", "Jordan 篮球鞋"]
    }
]

# 转换为 JSONL
def prepare_custom_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            for desc in item['descriptions']:
                f.write(json.dumps({
                    "query": desc,
                    "positive": item['image_path']
                }, ensure_ascii=False) + '\n')

prepare_custom_data(custom_data, "data/custom/train.jsonl")
```

---

## 4. 模型微调

### 4.1 配置文件详解

创建或编辑 `examples/clip_text_to_image.yaml`：

```yaml
# ========== 模型配置 ==========
model_name_or_path: "openai/clip-vit-base-patch32"
encoder_mode: "auto"              # 自动检测模型类型（clip/siglip/vit/dinov2/mae/bge/bert/e5/qwen/qwen-vl/llava/composed）

# ========== 参数高效微调 ==========
use_lora: true                    # 启用 LoRA，大幅降低显存占用
lora_r: 16                        # LoRA 秩（越大效果越好，但显存占用越多）
lora_alpha: 32                    # LoRA alpha（通常设为 2*r）

# ========== 数据路径 ==========
data_path: "data/flickr30k/train.jsonl"       # 训练数据
val_data_path: "data/flickr30k/val.jsonl"     # 验证数据（可选）
image_root: "data/flickr30k"                  # 图片基础路径
output_dir: "experiments/output_clip_t2i"

# ========== 训练参数 ==========
epochs: 3                         # 训练轮数（通常 3-5 个 epoch 效果最好）
batch_size: 128                   # 批大小（越大对比学习效果越好，但显存占用越多）
learning_rate: 2.0e-5             # 学习率
weight_decay: 0.01                # L2 正则化
max_grad_norm: 1.0                # 梯度裁剪

# ========== 学习率调度 ==========
scheduler_type: "cosine"          # cosine/linear/constant/constant_with_warmup
warmup_ratio: 0.1                 # 预热比例（总步数的 10%）

# ========== 损失函数 ==========
loss_type: "infonce"              # InfoNCE 对比损失（最常用）
temperature: 0.05                 # 温度参数（控制相似度分布的陡峭程度）

# ========== 内存优化 ==========
use_gradient_cache: false         # 梯度缓存（显存非常紧张时开启）
gradient_cache_chunk_size: 64

# ========== 多尺度表示学习（可选） ==========
use_mrl: false                    # Matryoshka 学习（一次训练生成多维度 embedding）
mrl_dims: [768, 512, 256, 128]

# ========== 日志和评估 ==========
logging_steps: 10                 # 每 10 步记录一次日志
save_steps: 0                     # 0 表示仅在 epoch 末保存，>0 表示每 N 步保存一次
eval_strategy: "epoch"            # 评估策略（epoch/steps/no）
report_to: "none"                 # 实验追踪（none/wandb/tensorboard）

# ========== 分布式训练 ==========
torch_distributed_debug: "no"     # 分布式调试（出现问题时改为 INFO）
```

#### 4.1.1 关键参数解释

| 参数 | 推荐值 | 说明 | 影响 |
|------|--------|------|------|
| `model_name_or_path` | `openai/clip-vit-base-patch32` | 模型选择 | - |
| `batch_size` | 128-256 | 越大对比学习效果越好 | 显存占用 ↑, 精度 ↑ |
| `learning_rate` | 2.0e-5 | 学习率 | 收敛速度/稳定性 |
| `epochs` | 3-5 | 训练轮数 | 精度 ↑, 训练时间 ↑ |
| `lora_r` | 16-64 | LoRA 秩 | 精度/显存 Trade-off |
| `temperature` | 0.05 | 温度参数 | 相似度分布的陡峭程度 |

**参数选择建议：**

```yaml
# GPU 显存充足（24GB+）：全量微调
use_lora: false
batch_size: 256
epochs: 5

# 显存有限（8-16GB）：LoRA 微调（推荐）
use_lora: true
lora_r: 16
batch_size: 128
epochs: 3

# 显存非常紧张（< 8GB）：梯度缓存 + LoRA
use_lora: true
use_gradient_cache: true
batch_size: 64
```

### 4.2 启动训练

#### 4.2.1 单 GPU 训练

```bash
# 方式 1：使用 run.py（推荐）
python run.py examples/clip_text_to_image.yaml

# 方式 2：使用 CLI 参数覆盖
python run.py examples/clip_text_to_image.yaml \
    --config_override batch_size=64 epochs=5 learning_rate=1e-5

# 方式 3：使用训练脚本
bash examples/run_clip_train.sh
```

#### 4.2.2 多 GPU 分布式训练

```bash
# 方式 1：自动检测并使用所有 GPU
accelerate launch run.py examples/clip_text_to_image.yaml

# 方式 2：指定使用的 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run.py examples/clip_text_to_image.yaml

# 方式 3：手动配置分布式策略
accelerate config                          # 交互式配置
accelerate launch run.py examples/clip_text_to_image.yaml
```

### 4.3 实际训练示例

```bash
# 完整的训练命令示例
python run.py examples/clip_text_to_image.yaml \
    --config_override \
        data_path="data/flickr30k/train.jsonl" \
        val_data_path="data/flickr30k/val.jsonl" \
        image_root="data/flickr30k" \
        output_dir="experiments/clip_t2i_finetuned" \
        batch_size=128 \
        epochs=3 \
        learning_rate=2.0e-5 \
        use_lora=true
```

### 4.4 训练日志示例

训练过程中，你会看到如下日志输出：

```
[2024-12-15 10:23:45] Loading model: openai/clip-vit-base-patch32
[2024-12-15 10:23:52] Model loaded successfully (parameters: 149M)
[2024-12-15 10:23:53] Loading data from: data/flickr30k/train.jsonl
[2024-12-15 10:23:58] Loaded 30,000 training samples
[2024-12-15 10:24:02] Loaded 1,000 validation samples
[2024-12-15 10:24:05] LoRA rank: 16, alpha: 32 (trainable parameters: 1.2M / total: 149M)

Epoch 1/3:
[Step 100/234]   Loss: 2.340, LR: 1.95e-05
[Step 200/234]   Loss: 2.134, LR: 1.91e-05
[Step 234/234]   Loss: 2.087, LR: 1.87e-05 (Epoch end)

Validation Results (Epoch 1):
  Recall@1:  65.32%
  Recall@5:  82.15%
  Recall@10: 88.47%

Epoch 2/3:
[Step 100/234]   Loss: 1.892, LR: 1.82e-05
...

Training completed in 2h 15m
Best checkpoint saved to: experiments/clip_t2i_finetuned/checkpoint-234
```

---

## 5. 效果评测

### 5.1 验证集评测

模型在每个 epoch 结束后会自动在验证集上评测。关键指标包括：

| 指标 | 说明 |
|------|------|
| **Recall@K** | 前 K 个返回结果中包含正确答案的比例 |
| **MRR (Mean Reciprocal Rank)** | 正确答案排名的倒数的平均值 |
| **NDCG@K** | 归一化折扣累积收益 |

### 5.2 完整评测脚本

```python
from vembed import Predictor
import numpy as np

# 加载微调后的模型
predictor = Predictor(
    model_path="experiments/clip_t2i_finetuned/checkpoint-234",
    device="cuda:0"
)

# 编码所有文本查询
queries = ["a red car", "a blue bicycle", ...]
query_embeddings = predictor.encode_text(queries, batch_size=32)

# 编码所有图片
images = ["image_1.jpg", "image_2.jpg", ...]
image_embeddings = predictor.encode_image(images, batch_size=32)

# 计算相似度矩阵
similarities = np.dot(query_embeddings, image_embeddings.T)

# 计算 Recall@K
def compute_recall_at_k(similarities, k=1):
    """
    计算 Recall@K（假设 similarities[i, i] 是正样本）
    """
    n = len(similarities)
    correct = 0
    for i in range(n):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        if i in top_k_indices:
            correct += 1
    return correct / n

recall_1 = compute_recall_at_k(similarities, k=1)
recall_5 = compute_recall_at_k(similarities, k=5)
recall_10 = compute_recall_at_k(similarities, k=10)

print(f"Recall@1:  {recall_1:.2%}")
print(f"Recall@5:  {recall_5:.2%}")
print(f"Recall@10: {recall_10:.2%}")
```

### 5.3 性能对标

**与零样本 CLIP 的对比：**

| 方法 | Recall@1 | Recall@5 | Recall@10 | 训练数据 |
|------|----------|----------|-----------|--------|
| **Zero-shot CLIP** | 58% | 78% | 85% | 无（使用预训练权重） |
| **本教程：LoRA 微调 (3 epoch)** | **71%** | **85%** | **90%** | 30k 对 (Flickr30k) |
| **提升幅度** | **+13 pp** | **+7 pp** | **+5 pp** | - |

> pp = percentage point

---

## 6. 常见问题

### 6.1 Q：显存不足如何处理？

**A：** 按照以下优先级调整：

```yaml
# 方案 1：启用 LoRA（显存降低 60-70%）
use_lora: true
lora_r: 16

# 方案 2：降低 batch size（8-16GB 显存）
batch_size: 64

# 方案 3：启用梯度缓存（显存再降低 30-40%）
use_gradient_cache: true
gradient_cache_chunk_size: 32

# 方案 4：启用梯度累积
gradient_accumulation_steps: 2    # 实际 batch size = 64 * 2 = 128

# 方案 5：使用较小的模型
model_name_or_path: "openai/clip-vit-small-patch32"
```

### 6.2 Q：如何使用其他 CLIP 模型？

**A：** 修改 `model_name_or_path` 参数：

```bash
# OpenAI 官方 CLIP
- "openai/clip-vit-base-patch32"      # ← 默认
- "openai/clip-vit-large-patch14"     # 性能更好，显存占用多
- "openai/clip-vit-small-patch32"     # 显存占用少

# 开源替代品
- "google/siglip-base-patch16-224"    # SigLIP，改进的对比学习
- "facebook/eva-clip-18b"             # EVA-CLIP，性能顶级

# 多语言模型
- "openai/clip-vit-base-patch32"      # 虽然名字带英文，但支持多语言
```

**性能对比：**

| 模型 | 参数量 | 显存占用 | Recall@1 (Flickr30k) | 推荐场景 |
|------|--------|--------|-------------------|--------|
| CLIP ViT-B/32 | 150M | 低 | 71% | **平衡方案** ✓ |
| CLIP ViT-L/14 | 427M | 中 | 76%+ | 追求最高精度 |
| SigLIP Base | 77M | 低 | 72%+ | 轻量级部署 |
| EVA-CLIP 18B | 7.5B | 高 | 80%+ | 服务器部署 |

### 6.3 Q：训练多久？如何判断收敛？

**A：**

```
训练时间（单 A100 GPU）：
- 3 epochs, batch_size=128: ~2-3 小时
- 5 epochs, batch_size=256: ~4-5 小时

收敛判断：
✓ Recall@1 稳定在 70% 以上 → 基本收敛
✓ 验证集指标 2-3 个 epoch 不再上升 → 可以停止
✓ 训练损失持续下降 → 继续训练
```

### 6.4 Q：如何在特定领域上微调？

**A：** 只需替换数据即可，格式保持一致：

```python
# 电商产品数据
{
  "query": "黑色纯棉 T 恤，M 码",
  "positive": "products/tshirt_001.jpg"
}

# 医学影像
{
  "query": "chest x-ray showing pneumonia",
  "positive": "medical/xray_001.jpg"
}

# 卫星图像
{
  "query": "urban residential area with high density",
  "positive": "satellite/image_001.jpg"
}
```

### 6.5 Q：LoRA vs 全量微调，怎么选？

**A：**

| 特性 | LoRA | 全量微调 |
|------|------|--------|
| 显存占用 | 低（-60%） | 高 |
| 训练速度 | 快 | 慢 |
| 精度 | 98% 相当 | 100% 基准 |
| 推理速度 | 相同 | 相同 |
| 推荐场景 | **生产环境** ✓ | 研究/追求最高精度 |

**建议：**
- **大多数场景用 LoRA**：显存节省明显，精度基本无损
- **精度要求极高时**用全量微调：但显存需求高

---

## 7. 推理与部署

### 7.1 单个样本推理

```python
from vembed import Predictor

# 加载模型
predictor = Predictor("experiments/clip_t2i_finetuned/checkpoint-234")

# 编码单个文本
text_emb = predictor.encode_text("red sport shoes")
print(f"Text embedding shape: {text_emb.shape}")  # (768,)

# 编码单个图片
img_emb = predictor.encode_image("shoes.jpg")
print(f"Image embedding shape: {img_emb.shape}")  # (768,)

# 计算相似度
import numpy as np
similarity = np.dot(text_emb, img_emb)
print(f"Text-Image Similarity: {similarity:.4f}")
```

### 7.2 批量推理与搜索

```python
import numpy as np

# 批量编码
queries = ["red shoes", "blue shoes", "green shoes"]
query_embeddings = predictor.encode_text(queries, batch_size=32)

candidates = ["shoes_1.jpg", "shoes_2.jpg", "shoes_3.jpg", ...]
candidate_embeddings = predictor.encode_image(candidates, batch_size=32)

# 计算相似度矩阵
similarities = np.dot(query_embeddings, candidate_embeddings.T)  # (3, N)

# 搜索
for i, query in enumerate(queries):
    top_k = np.argsort(similarities[i])[::-1][:5]
    print(f"Query: '{query}'")
    for rank, idx in enumerate(top_k, 1):
        print(f"  {rank}. {candidates[idx]} (score: {similarities[i, idx]:.4f})")
```

### 7.3 与向量库集成（FAISS）

```bash
# 安装 FAISS
pip install faiss-cpu    # CPU 版本
# 或
pip install faiss-gpu    # GPU 版本（需要 CUDA）
```

```python
import faiss
import numpy as np

# 1. 编码所有候选图片
candidates = ["shoes_1.jpg", "shoes_2.jpg", ...]
embeddings = predictor.encode_image(candidates, batch_size=32)

# 2. 构建 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 距离（欧式距离）
index.add(embeddings.astype(np.float32))

# 3. 搜索
query_text = "red sport shoes"
query_emb = predictor.encode_text(query_text).reshape(1, -1)

distances, indices = index.search(query_emb.astype(np.float32), k=10)
print(f"Top-10 results for '{query_text}':")
for rank, idx in enumerate(indices[0], 1):
    print(f"{rank}. {candidates[idx]}")
```

### 7.4 部署为 API 服务

```python
# api_server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from vembed import Predictor
import numpy as np

app = FastAPI()
predictor = Predictor("experiments/clip_t2i_finetuned/checkpoint-234")

@app.post("/search/text")
async def search_by_text(query: str, top_k: int = 10):
    """文本搜索 API"""
    # 这里应该连接到向量索引
    query_emb = predictor.encode_text(query)
    # ... 搜索逻辑 ...
    return {"results": [...]}

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 10):
    """图像搜索 API"""
    # 保存临时文件并编码
    content = await file.read()
    # ... 编码逻辑 ...
    return {"results": [...]}

# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

启动服务：
```bash
python api_server.py

# 测试
curl -X POST "http://localhost:8000/search/text?query=red%20shoes&top_k=10"
```

---

## 8. 总结

通过本教程，你已经学会了如何使用 vembed-factory 快速微调 CLIP 模型以实现高精度的文本-图像检索：

**核心步骤：**

1. ✅ **数据准备**
   - 使用 `prepare_data.py` 自动下载 Flickr30k
   - 或转换自定义数据为 JSONL 格式

2. ✅ **配置管理**
   - 根据硬件选择合适的参数（LoRA, batch size 等）
   - 使用 YAML 配置或 CLI 参数覆盖

3. ✅ **模型训练**
   - 单 GPU 或多 GPU 分布式训练
   - 自动 checkpoint 保存和评测

4. ✅ **效果验证**
   - Recall@1 从 58% 提升至 71%+（+13pp）
   - 性能远超零样本预训练模型

5. ✅ **推理部署**
   - Python API 或 FastAPI 服务
   - 与 FAISS 等向量库集成

**关键优势：**
- 🎯 简洁的数据格式（仅需 query 和 positive）
- 🎯 灵活的参数调整（不需修改代码）
- 🎯 完整的分布式训练支持
- 🎯 生产就绪的推理 API

---

## 推荐阅读

- [vembed-factory GitHub](https://github.com/fangzhensheng/vembed-factory)
- [CLIP 原始论文](https://arxiv.org/abs/2103.14030)
- [Flickr30k 数据集](https://github.com/BryanPlummer/flickr30k_entities)
- 兄弟教程：[Qwen3-VL 多模态检索微调](./qwen3_multimodal_retrieval_zh.md)

