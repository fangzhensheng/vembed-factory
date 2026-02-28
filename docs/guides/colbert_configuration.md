# ColBERT 配置指南

## 什么是 ColBERT？

**ColBERT (Contextualized Late Interaction)** 是一种**后期交互检索架构**，它改变了计算相似度的方式：

### vs. 传统密集检索

```
传统密集检索:  文本/图像 → [单个向量 D维] → 余弦相似度
ColBERT:      文本/图像 → [所有token的向量 L×D] → MaxSim相似度
```

**优势：**
- ✅ **精细搜索**：逐 token 比较，捕捉细节而非全局特征
- ✅ **内存高效**：Gradient Cache + LoRA 支持大批量训练
- ✅ **灵活修剪**：可选的注意力引导 token 选择，优化内存/速度

---

## ColBERT 配置要素（两个参数）

### 1️⃣ Loss Function

```yaml
loss_type: colbert
```

**含义：** 使用 **MaxSim** 评分机制而不是全局余弦相似度

**公式：**
```
Score(Q, D) = mean_q ( max_d (q_i · d_j) )
```

- `q_i` = 查询中第 i 个 token 的嵌入
- `d_j` = 文档中第 j 个 token 的嵌入
- 对每个查询 token，计算其与所有文档 token 的最大相似度
- 最后对所有查询 token 取平均

**代码位置：** [vembed/losses/functions/colbert.py](../../vembed/losses/functions/colbert.py)

---

### 2️⃣ Embedding Output Format

```yaml
pooling_method: none
```

**含义：** 返回**所有 token 的向量**而非单个向量

| 参数值 | 输出形状 | 用途 |
|--------|---------|------|
| `none` | `[B, L, D]` | **ColBERT** — 所有 L 个 tokens |
| `cls` | `[B, D]` | 传统 — [CLS] token 只 |
| `mean` | `[B, D]` | 传统 — 平均池化 |
| `last_token` | `[B, D]` | 传统 — 最后一个 token |

**必需条件：** ColBERT 必须使用 `pooling_method: none`（保留所有 token 用于逐 token 比较）

设置 `loss_type: colbert` 时，`pooling_method` 会自动设为 `none`，也可显式指定。

**不同模型的行为：**
- **Vision (DINOv2)**：返回 `[B, 257, D]` — 1 个 CLS token + 256 个 patch tokens
- **VLM (Qwen)**：返回 `[B, seq_len, D]` — 所有生成 tokens

---

### 3️⃣ Token 优化（可选）

```yaml
topk_tokens: 32    # 可选参数
```

**含义：** 启用注意力引导的 token 修剪（**优化而非必需**）

**工作原理：**
1. 计算 [CLS] token 与各 patch 的**余弦相似度**
2. 选择相似度最高的 K 个 patch
3. 保留这 K 个 patch + CLS token = `[B, K+1, D]`

**参数值：**
- `topk_tokens: 0` — 保留所有 tokens（默认）
- `topk_tokens: 32` — 保留前 32 个最相关 patches（推荐用于 Vision）
- `topk_tokens: 64` — 保留前 64 个（更多细节）

**优化效果：**
- 减少 MaxSim 计算量（O(L²) → O(K·L)）
- 内存消耗降低
- **准确率几乎不变**（对象中心化的 attention 已过滤噪声）

**代码位置：** [vembed/model/backbones/auto.py:163-208](../../vembed/model/backbones/auto.py#L163-L208)

---

## 完整配置模板

### 最小配置（必需）

```yaml
loss_type: colbert
pooling_method: none
```

### 推荐完整配置

```yaml
# ===== REQUIRED =====
loss_type: colbert           # MaxSim loss
pooling_method: none         # Token-level embeddings (auto-set when loss_type=colbert)

# ===== OPTIONAL: Optimization =====
topk_tokens: 32              # Attention-guided token pruning (0=disabled)
projection_dim: 128          # Embedding dimension reduction

# ===== STANDARD TRAINING =====
epochs: 20
batch_size: 128
learning_rate: 5e-5

# ===== MEMORY OPTIMIZATION (推荐) =====
use_gradient_cache: true     # 大批量支持
use_lora: true               # 参数高效微调

# ===== DATA & PATHS =====
data_path: data/train.jsonl
val_data_path: data/val.jsonl
output_dir: experiments/my_colbert
```

---

## ColBERT vs. 其他检索方式对比

| 方式 | Loss | Pooling | Embedding形状 | 用途 | 优点 | 缺点 |
|------|------|---------|--------------|------|------|------|
| **密集检索** | InfoNCE | mean/cls | `[B, D]` | 通用 | 快速、简单 | 信息丢失 |
| **ColBERT** | colbert | **none** | **[B, L, D]** | **精细搜索** | **精准、可解释** | **内存占用** |
| **MRL** | mrl | mean/cls | `[B, D]` | 多粒度 | 灵活 | 复杂 |
| **Triplet** | triplet | cls/mean | `[B, D]` | 简单对比 | 稳定 | 准确度低 |

---

## 常见配置示例

### 示例 1: DINOv2 + ColBERT（图像检索）

```yaml
# Model
model_name: facebook/dinov2-base
retrieval_mode: i2i

# ColBERT
loss_type: colbert
pooling_method: none
topk_tokens: 32            # DINOv2 有 256 个 patches，保留前 32 个

# Embedding
projection_dim: 128        # 可选

# Training
epochs: 20
batch_size: 128
learning_rate: 5e-5
use_gradient_cache: true
use_lora: true
```

**说明：**
- `topk_tokens: 32` 将 257 个 tokens（1 CLS + 256 patches）减少到 33 个
- 计算量减少 ~7x，几乎不影响准确率
- 推荐用于 vision-only 模型

---

### 示例 2: Qwen + ColBERT（文本检索）

```yaml
# Model
model_name: Qwen/Qwen2-7B-Instruct
retrieval_mode: t2t

# ColBERT
loss_type: colbert
pooling_method: none
# 注意：VLM 的序列长度可变（512-8192），不建议用 topk_tokens

# Embedding
projection_dim: 128

# Training
epochs: 3
batch_size: 64
learning_rate: 5e-5
use_gradient_cache: true
use_lora: true
```

**说明：**
- VLM 的 token 已是高层表示，通常不需要 token 修剪
- 可变序列长度使得固定 `topk_tokens` 效果不稳定
- 优先用 Gradient Cache 和 LoRA 优化内存

---

### 示例 3: Qwen + ColBERT + TopK（多模态检索）

```yaml
model_name: Qwen/Qwen2-VL-32B-Instruct
retrieval_mode: m2i

loss_type: colbert
pooling_method: none
topk_tokens: 64            # 可选：保留前 64 个最相关 tokens

projection_dim: 256
use_mrl: false             # ColBERT 不支持 MRL

# Training
epochs: 10
batch_size: 32
learning_rate: 2e-5
use_gradient_cache: true
```

---

## 配置验证

### ✅ 验证方法 1: 运行 Dry Run

```bash
python run.py examples/dinov2_colbert.yaml --dry_run
```

生成的配置文件应包含：
```yaml
loss_type: colbert
pooling_method: none
topk_tokens: 32
```

### ✅ 验证方法 2: 检查训练配置输出

```bash
cat experiments/output_sop_dinov2_colbert/.train_config.yaml | grep -E "^(loss_type|pooling_strategy|topk_tokens):"
```

应输出：
```
loss_type: colbert
pooling_method: none
topk_tokens: 32
```

### ✅ 验证方法 3: 检查模型推理配置

```bash
cat experiments/output_sop_dinov2_colbert/checkpoint-*/vembed_config.json | python -m json.tool | grep -E "pooling|topk"
```

应包含：
```json
{
  "pooling_method": "none",
  "topk_tokens": 32,
  ...
}
```

---

## 常见错误和解决方案

### ❌ 错误 1: 混淆 Loss 和 Pooling

```yaml
# 错误：
loss_type: colbert
pooling_method: colbert   # ← 这个不对
```

**原因：** `pooling_method: colbert` 是 VLM 特定的，不是通用的 ColBERT pooling

**解决：**
```yaml
# 正确：
loss_type: colbert
pooling_method: none      # ← 所有 ColBERT 都用这个
```

---

### ❌ 错误 2: 遗漏 pooling_strategy

```yaml
# 可以工作，但隐式依赖（不推荐）：
loss_type: colbert
# 缺少 pooling_strategy
```

**说明：** CLI 会自动设置为 `none`，但用户看不到

**解决：**
```yaml
# 推荐：显式指定（清晰、可读）
loss_type: colbert
pooling_method: none
```

---

### ❌ 错误 3: 与 MRL 混用

```yaml
# 错误：不支持
loss_type: colbert
use_mrl: true              # ← ColBERT 不支持 MRL
```

**原因：**
- ColBERT 需要 token-level 表示 `[B, L, D]`
- MRL 需要多层 pooling 后的向量 `[B, dims[], D]`
- 两者互不兼容

**解决：** 二选一
```yaml
# 方案 A: ColBERT
loss_type: colbert
pooling_method: none
use_mrl: false

# 方案 B: MRL
loss_type: infonce
use_mrl: true
pooling_method: mean
```

---

### ❌ 错误 4: topk_tokens 设置不当

```yaml
# 不推荐 1: 过小（丢失信息）
topk_tokens: 4             # 太小

# 不推荐 2: 超过总 tokens
topk_tokens: 1000          # 超过 DINOv2 的 257 个 tokens
```

**建议值：**
| 模型 | 总 tokens | 建议 topk_tokens |
|------|---------|----------------|
| DINOv2-base | 257 | 32, 64 |
| DINOv2-large | 577 | 64, 128 |
| Qwen-VL | 可变 | 0（不修剪）|
| ResNet50 | ~50 | 16, 32 |

---

### ⚠️ 警告: topk_tokens 无效

如果 `topk_tokens` 设置了但没有生效：

```bash
# 检查 pooling_strategy
grep "pooling_method:" .train_config.yaml
```

如果是 `pooling_method: mean` 或其他非 `none` 的值，`topk_tokens` 会被忽略（因为已经全局池化）

**解决：**
```yaml
pooling_method: none     # 必需，才能使用 topk_tokens
topk_tokens: 32
```

---

## 性能指标和优化

### 内存占用对比

```
传统密集检索：  [B, D] = [128, 768] = 98 KB
ColBERT:      [B, L, D] = [128, 257, 768] = 25 MB （26x 增长）
+ topk_tokens:  [B, K+1, D] = [128, 33, 768] = 3.2 MB （仅 3.3x 增长）
```

### 计算成本对比

```
相似度计算：
- 传统：O(B·D) — 向量点积
- ColBERT：O(B·L_q·L_d·D) — token-level 逐一比较
- + topk_tokens：O(B·K·L_d·D) — 大幅减少

MaxSim 论文报告：topk_tokens=32 时，速度 5-7x 加快，准确率几乎不变
```

### 优化建议

| 场景 | 推荐配置 |
|------|---------|
| **小数据集** (< 10K docs) | `topk_tokens: 0` — 精准优先 |
| **中等数据集** (10K - 100K) | `topk_tokens: 32` — 平衡 |
| **大数据集** (> 100K) | `topk_tokens: 64` + `use_gradient_cache: true` |
| **内存受限** (< 24 GB) | `topk_tokens: 16`, `batch_size: 32`, `gradient_cache_chunk_size: 4` |

---

## 参考资源

- **ColBERT 原论文：** [SIGIR 2020 - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction Retrieval](https://arxiv.org/abs/2004.12832)
- **MaxSim 评分实现：** [vembed/losses/functions/colbert.py](../../vembed/losses/functions/colbert.py) — `_maxsim()` 函数
- **Token 修剪算法：** [vembed/model/backbones/auto.py](../../vembed/model/backbones/auto.py) — `_attention_topk_tokens()` 方法
- **完整示例：** [examples/dinov2_colbert.yaml](../examples/dinov2_colbert.yaml) 和 [examples/qwen_colbert.yaml](../examples/qwen_colbert.yaml)

---

## 常见问题 FAQ

**Q: ColBERT 比传统方法快吗？**

A: 单次查询**更慢**（需逐 token 比较），但准确率更高。适合小数据集或实时性不是瓶颈的场景。加上 `topk_tokens` 修剪后，速度接近传统方法但准确率更好。

---

**Q: 能和 LoRA 一起用吗？**

A: ✅ 可以，推荐使用。配置：`use_lora: true`

---

**Q: 能和 Gradient Cache 一起用吗？**

A: ✅ 可以，推荐使用。配置：`use_gradient_cache: true` 支持大批量。

---

**Q: topk_tokens 影响准确率吗？**

A: 几乎没有。论文表明 topk_tokens=32 时准确率下降 < 1%，但速度 5-7x 提升。

---

**Q: 可以在多个 GPU 上训练吗？**

A: ✅ 可以。使用：`accelerate launch` 或指定 `num_gpus` 参数。

---

**Q: 推理时需要特殊处理吗？**

A: ✅ 需要。推理时仍需返回 token-level 嵌入 `[B, L, D]`，然后用 MaxSim 计算相似度。参见 [vembed/inference.py](../../vembed/inference.py)
