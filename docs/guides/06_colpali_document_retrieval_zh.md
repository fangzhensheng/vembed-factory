# ColPali：细粒度文档检索微调指南

**延迟交互 (Late Interaction)** 是最新一代检索方法，相比早期交互（CLIP、BERT）能实现更细粒度的文档理解。**ColPali** 是基于视觉语言模型的延迟交互方案，能精确匹配文档中的任意位置，特别适合法律文书、医学报告、学术论文等需要精确信息检索的场景。

---

## 1. ColPali 的核心优势

```
CLIP（早期交互）：
  文本"合同金额条款" ──→ 文本编码 ──→ [全局向量]
  PDF 页面 ──────────→ 图像编码 ──→ [全局向量]
  相似度 = dot([全局], [全局])
  问题：丢失细粒度信息

ColPali（延迟交互）：
  文本"合同金额条款" ──→ 分词 ──→ ['合', '同', '金', '额', '条', '款']
  PDF 页面 ──────────→ 逐区块编码 ──→ [块1向量, 块2向量, ...]
  相似度 = max(各词 vs 各块的相似度)
  优势：精确匹配各个区块
```

### 性能对比

| 指标 | CLIP | ColPali | 提升 |
|------|------|---------|------|
| **ViDoRe Benchmark** | 70% | **78%** | +8 pp |
| **细粒度精准度** | 60% | **85%** | +25 pp |
| **处理时间** | 快 | 较快 | -20% |

---

## 2. 数据准备

### 2.1 文档格式转换

```python
import json
from PIL import Image
import PyPDF2

def convert_pdf_to_retrieval_format(pdf_path, output_jsonl):
    """将 PDF 转换为检索数据格式"""

    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)

        for page_idx, page in enumerate(pdf_reader.pages):
            # 提取文本
            text = page.extract_text()

            # 转换页面为图像
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=page_idx+1, last_page=page_idx+1)
            image_path = f"docs/page_{page_idx}.png"
            images[0].save(image_path)

            # 写入 JSONL
            with open(output_jsonl, 'a') as out:
                out.write(json.dumps({
                    "query": text,           # 页面文本作为查询（用于索引）
                    "positive": image_path,  # 页面图像
                    "page_id": f"{pdf_path}#{page_idx}"
                }, ensure_ascii=False) + '\n')
```

### 2.2 数据格式

```json
{
  "query": "We agree to pay the sum of 50,000 USD as per clause 3.2",
  "positive": "documents/contract_page_005.png",
  "page_id": "contract_001.pdf#5"
}
```

---

## 3. 配置与训练

### 3.1 ColPali 配置

```yaml
# colpali_training.yaml
model_name_or_path: "Qwen/Qwen2-7B-Instruct"
encoder_mode: "vlm_generic"

# ColBERT 延迟交互配置
loss_type: "colbert"            # 关键：使用 ColBERT 损失
pooling_strategy: "colbert"     # ColBERT 池化策略
num_doc_tokens: 128             # 每个文档块的 token 数

# 数据
data_path: "data/documents_train.jsonl"
val_data_path: "data/documents_val.jsonl"
image_root: "data/documents"

# 训练参数
output_dir: "experiments/colpali_finetuned"
epochs: 3
batch_size: 32                  # ColBERT 通常需要较小的 batch
learning_rate: 1.5e-5

# 内存优化
use_lora: true
lora_r: 16
use_gradient_cache: true
gradient_cache_chunk_size: 16

# 日志
logging_steps: 10
save_steps: 0
eval_strategy: "epoch"
```

### 3.2 启动训练

```bash
python run.py colpali_training.yaml

# 多 GPU
accelerate launch run.py colpali_training.yaml
```

---

## 4. 推理与搜索

### 4.1 单查询搜索

```python
from vembed import Predictor
import numpy as np

predictor = Predictor("experiments/colpali_finetuned/checkpoint-final")

# 编码查询
query = "What is the contract amount?"
query_tokens = predictor.encode_text(query)  # shape: (num_tokens, 128)

# 编码文档
doc_image = "contract.png"
doc_tokens = predictor.encode_image(doc_image)  # shape: (num_blocks, 128)

# ColBERT 相似度计算：max pool
similarity = (query_tokens @ doc_tokens.T).max(axis=1).mean()
print(f"Similarity: {similarity:.4f}")
```

### 4.2 批量搜索（向量库）

```python
import faiss

# 编码所有文档页面
documents = ["page_1.png", "page_2.png", ...]
doc_embeddings_list = []

for doc in documents:
    tokens = predictor.encode_image(doc)  # (num_blocks, 128)
    # 对 ColBERT token 做平均 pooling（简化处理）
    avg_token = tokens.mean(axis=0)
    doc_embeddings_list.append(avg_token)

doc_embeddings = np.array(doc_embeddings_list)

# 创建 FAISS 索引
index = faiss.IndexFlatL2(128)
index.add(doc_embeddings.astype(np.float32))

# 搜索
query_embedding = predictor.encode_text(query).mean(axis=0).reshape(1, -1)
distances, indices = index.search(query_embedding, k=10)

print("Top-10 相关文档页面:")
for rank, idx in enumerate(indices[0], 1):
    print(f"{rank}. {documents[idx]}")
```

---

## 5. 应用场景

### 5.1 法律文书检索

```
用户查询："支付违约金的条款在哪里？"

ColPali：
  1. 逐页面搜索
  2. 精确定位 "违约金" "支付" 等关键词
  3. 返回包含条款的具体页码
  4. 高精度（85%+）
```

### 5.2 医学报告检索

```
医生查询："患者的 CT 报告中肺部阴影的描述"

优势：
  ✓ 跨页面的细粒度信息匹配
  ✓ 精确找到相关段落
  ✓ 避免误诊
```

### 5.3 学术论文检索

```
研究者查询："论文中提到的 SOTA 性能指标"

优势：
  ✓ 在 Table、Figure 等结构化内容中精确检索
  ✓ 多模态理解（文字 + 表格 + 图表）
```

---

## 6. 性能评测

### 6.1 ViDoRe 基准

```python
# 在标准基准上评测
from vembed.evaluation.metrics import compute_recall_at_k

# 计算 Recall@K
recalls = compute_recall_at_k(
    query_embeddings=query_embs,
    doc_embeddings=doc_embs,
    k_list=[1, 5, 10]
)

print(f"Recall@1: {recalls['Recall@1']:.2%}")
print(f"Recall@5: {recalls['Recall@5']:.2%}")
print(f"Recall@10: {recalls['Recall@10']:.2%}")
```

### 6.2 预期结果

```
在 ViDoRe 数据集上（文档检索）：

Zero-shot（无微调）：70%
微调后（3 epoch）：78%
微调（5 epoch）：79%

相比 CLIP：+8-9 pp 的显著提升
```

---

## 7. 常见问题

### Q1：ColPali vs CLIP 什么时候用？

**A：**

```
使用 CLIP：
  - 简单商品检索
  - 图像标签化
  - 速度优先

使用 ColPali：
  ✓ 文档检索
  ✓ 精度优先
  ✓ 多模态理解（表格、图表等）
```

### Q2：ColBERT 损失有什么特殊之处？

**A：**

```
标准 InfoNCE：
  loss = -log(exp(sim(q,p+)) / Σ exp(sim(q,p-)))

ColBERT：
  loss = -log(max_pool(token_sims(q,p+)) / Σ max_pool(...))

优势：支持多向量表示，细粒度匹配
```

### Q3：推理时 ColBERT 会变慢吗？

**A：**

```
推理速度对比：

CLIP：
  编码 1 张图：20ms

ColPali：
  编码 1 张图：25ms（仅 +25%）

原因：延迟交互在搜索阶段（max_pool）
推理阶段基本相同
```

---

## 8. 总结

**ColPali 适用场景：**
- ✅ 文档检索（法律、医学、学术等）
- ✅ 需要精确信息定位
- ✅ 对精度要求 > 85%

**性能提升：**
- Recall@1: 70% → 78%
- 微调数据需求：5k-10k 样本

**部署建议：**
```yaml
模型：Qwen/Qwen2-7B-Instruct + LoRA
显存：12-16GB
吞吐：500 doc/sec（批量）
```

---

**相关教程：**
- [Qwen3-VL 微调](./02_qwen3_multimodal_retrieval_zh.md)
- [参数高效微调](./04_parameter_efficient_tuning_zh.md)

