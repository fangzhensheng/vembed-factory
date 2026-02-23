# Gradient Cache — 详细技术文档

## 1. 概述

Gradient Cache 是一种**显存高效的大 batch 对比学习训练技术**。它解耦了前向传播的显存消耗与 loss 计算所需的 batch size，使得在有限 GPU 显存下也能使用大 batch（例如 batch_size=128）进行 InfoNCE 等对比损失的训练。

本项目的实现分为两层：

| 层级 | 文件 | 职责 |
|------|------|------|
| **底层库** | `vembed/grad_cache/grad_cache.py` | 通用 Gradient Cache 算法，模型无关 |
| **上层封装** | `vembed/training/gradient_cache.py` | vembed 特定的 batch 拆包、VLM 输入分割、混合精度适配 |

---

## 2. 核心原理

### 2.1 问题背景

对比学习（如 InfoNCE）的效果高度依赖 in-batch negatives 的数量。batch_size 越大，负样本越多，loss 越有效。但大 batch 的前向传播需要大量显存，尤其是 VLM（视觉语言模型）每张图片可能占几百 MB。

**矛盾**：loss 需要大 batch，但显存只够小 batch 前向。

### 2.2 Gradient Cache 的三步法

Gradient Cache 通过「先收集表示，再计算梯度，最后回传」来解决这个矛盾：

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Forward (no grad) — 分 chunk 前向，收集 representations │
│                                                               │
│  batch (B=32)  ─split─►  chunk_1 (2) → rep_1                │
│                           chunk_2 (2) → rep_2                │
│                           ...                                │
│                           chunk_16(2) → rep_16               │
│                                                               │
│  concat ──► all_reps: [32, dim]   (占显存很少，只是向量)        │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Build Cache — 用全部 representations 算 loss + 梯度    │
│                                                               │
│  loss = InfoNCE(query_reps[32], positive_reps[32])           │
│  loss.backward()  ──► grad_cache: dL/d(reps)                │
│                                                               │
│  (只对 representation 向量求梯度，不涉及模型参数，开销极小)       │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Forward-Backward — 分 chunk 重新前向 + 用缓存梯度反传  │
│                                                               │
│  for each chunk_i:                                           │
│    rep_i = model(chunk_i)           # 重新前向                │
│    surrogate = dot(rep_i, grad_i)   # 代理标量               │
│    surrogate.backward()             # 梯度传到模型参数         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数学原理

设模型参数为 θ，输入 x，模型输出表示 r = f_θ(x)，损失 L(r_1, r_2, ...)。

由链式法则：

\[
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial \theta}
\]

Gradient Cache 分两步计算：

1. **Build Cache**: 计算 ∂L/∂r（只需 representations，不需要模型前向图）
2. **Forward-Backward**: 对每个 chunk 计算 ∂r/∂θ，用代理损失 `surrogate = dot(r, ∂L/∂r)` 反传

代理损失的梯度恰好是：

\[
\frac{\partial}{\partial \theta} \text{dot}(r, g) = g \cdot \frac{\partial r}{\partial \theta} = \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial \theta} = \frac{\partial L}{\partial \theta}
\]

因此数学上完全等价。

### 2.4 RandContext — 随机状态保存

Dropout 等随机操作要求两次前向传播的随机状态一致（否则梯度不对）。`RandContext` 在第一次前向时保存 CPU + GPU 的 RNG 状态，第二次前向时恢复：

```python
# 第一次前向（Step 1）
rnd_state = RandContext(*tensors)   # 保存当前随机状态
output = model(chunk)               # dropout 等使用当前随机数

# 第二次前向（Step 3）
with rnd_state:                     # 恢复保存的随机状态
    output = model(chunk)           # dropout 使用相同的随机数 → 输出一致
```

---

## 3. 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 每 GPU 的**总** batch size，决定 InfoNCE 的 logits 矩阵大小 [B×B] |
| `gradient_cache_chunk_size` | 4 | 每次前向的 micro-batch 大小，决定**显存**消耗 |
| `use_gradient_cache` | true | 是否启用 Gradient Cache |

### 3.1 batch_size 与 gradient_cache_chunk_size 的关系

```
batch_size = 32, gradient_cache_chunk_size = 2

DataLoader 输出 32 个样本
    │
    ├──► 分成 16 个 chunk，每个 2 个样本
    │    chunk_1: model(2 samples)   ← 只占 2 个样本的显存
    │    chunk_2: model(2 samples)
    │    ...
    │    chunk_16: model(2 samples)
    │
    ├──► 收集 32 个 representations
    │
    └──► InfoNCE loss 在 32×32 logits 矩阵上计算
         （等价于 batch_size=32 的效果，但显存只需 2 个样本）
```

### 3.2 调参建议

| 场景 | batch_size | chunk_size | 说明 |
|------|-----------|------------|------|
| CLIP (ViT-B/32) | 128 | 32 | 图片小、模型轻，可以大 chunk |
| Qwen3-VL-2B | 32 | 2 | VLM 单张图占显存大，chunk 要小 |
| Qwen3-VL-8B | 16-32 | 1-2 | 更大模型，chunk 更小 |
| SigLIP | 64-128 | 16-32 | 类 CLIP 架构 |

**关键原则**：
- `batch_size` 越大越好（更多 in-batch negatives）
- `gradient_cache_chunk_size` 越小显存越省，但训练越慢（更多 forward 次数）
- 如果 `batch_size == gradient_cache_chunk_size`，则 Gradient Cache 退化为普通训练

### 3.3 常见问题：loss 接近 0

如果 loss 接近 0（如 0.0001），通常是 `batch_size` 太小。例如：

- `batch_size=2` → logits 矩阵只有 2×2，模型只需从 1 个负样本中区分正样本 → 太简单
- `batch_size=32` → logits 矩阵 32×32，模型需从 31 个负样本中区分 → 有效训练

---

## 4. 架构与代码结构

### 4.1 底层库 — `GradCache`

```
vembed/grad_cache/
├── __init__.py
├── grad_cache.py         # 核心 GradCache 类
└── context_managers.py   # RandContext 随机状态管理
```

`GradCache` 类的核心方法：

| 方法 | 功能 |
|------|------|
| `split_inputs()` | 将输入分成 chunks（支持 Tensor、dict、list） |
| `model_call()` | 根据输入类型调用模型（`model(**dict)` / `model(*list)` 等） |
| `get_reps()` | 从模型输出中提取 representation（支持 Tensor、tuple、ModelOutput） |
| `forward_no_grad()` | Step 1：无梯度前向，收集所有 chunk 的 representations + 随机状态 |
| `build_cache()` | Step 2：用全部 representations 计算 loss，得到 dL/d(reps) |
| `forward_backward()` | Step 3：用缓存梯度对每个 chunk 做前向+反传 |
| `cache_step()` | 组合以上三步的完整训练步骤 |

### 4.2 上层封装 — `GradientCache`

```
vembed/training/gradient_cache.py
├── _extract_rep()        # 从各类模型输出提取表示向量
├── _split_vlm_inputs()   # VLM 专用的输入分割函数（核心扩展）
└── GradientCache         # 封装类
    ├── _unpack_batch()   # 从 collator 输出拆分 query/positive/negative
    └── step()            # 执行一个完整训练步骤
```

### 4.3 数据流全景

```
                         Collator (qwen.py / default.py)
                                    │
                         batch_output (dict)
                         ┌─────────────────────────┐
                         │ query_input_ids          │
                         │ query_attention_mask     │
                         │ query_pixel_values       │
                         │ query_image_grid_thw     │
                         │ pos_input_ids            │
                         │ pos_attention_mask       │
                         │ pos_pixel_values         │
                         │ pos_image_grid_thw       │
                         │ (neg_pixel_values)       │
                         │ (neg_image_grid_thw)     │
                         └───────────┬─────────────┘
                                     │
                          GradientCache._unpack_batch()
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              q_batch            p_batch          n_batch
            {input_ids,      {input_ids,       {pixel_values,
             attn_mask}       attn_mask,         image_grid_thw,
                              pixel_values,      input_ids, ...}
                              image_grid_thw}
                    │                │                │
                    ▼                ▼                ▼
              _split_vlm_inputs() 分 chunk
                    │                │                │
                    ▼                ▼                ▼
              GradCache.cache_step(q_chunks, p_chunks, [n_chunks])
                    │
                    ▼
                loss (scalar)
```

---

## 5. VLM 兼容性 — `_split_vlm_inputs`

### 5.1 问题：VLM 的 tensor 维度不一致

对于传统模型（CLIP/SigLIP），所有 tensor 的 dim=0 都是 batch_size：

```
input_ids:      [B, seq_len]
attention_mask: [B, seq_len]
pixel_values:   [B, C, H, W]       ← dim=0 = B
```

但对于 Qwen3-VL 等 VLM：

```
input_ids:      [B, seq_len]        ← dim=0 = B
attention_mask: [B, seq_len]        ← dim=0 = B
pixel_values:   [total_patches, C, patch_H, patch_W]  ← dim=0 ≠ B !!
image_grid_thw: [num_images, 3]     ← dim=0 = B (假设每样本一张图)
```

`pixel_values` 是所有图片的 patch 拍平后的总量（每张图的 patch 数量不同，取决于图片分辨率）。如果直接按 `chunk_size` 在 dim=0 split，会把 `pixel_values` 错误地切割。

### 5.2 解决方案

`_split_vlm_inputs` 函数根据 `image_grid_thw` 计算每张图的 patch 数，正确分割 `pixel_values`：

```python
# 每张图的 patch 数 = t × h × w
patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

# chunk_i 的 pixel_values = 该 chunk 对应图片的所有 patches
chunk_n_patches = sum(patches_per_image[start:end])
chunk_dict["pixel_values"] = pixel_values[px_offset : px_offset + chunk_n_patches]
```

### 5.3 对不同模型的处理

`_split_vlm_inputs` 自动检测输入类型并选择分割策略：

| 输入类型 | 检测条件 | 分割策略 |
|----------|---------|---------|
| 纯文本（query） | 无 `pixel_values` 或 `image_grid_thw` | 标准 dim=0 split |
| CLIP/SigLIP 图片 | `pixel_values` 存在但无 `image_grid_thw` | 标准 dim=0 split（dim=0 = B） |
| Qwen3-VL 图片 | 同时有 `pixel_values` 和 `image_grid_thw` | VLM-aware split（按 patch 数量） |
| 纯 Tensor | `isinstance(input, Tensor)` | 标准 `tensor.split()` |

---

## 6. Batch 拆包逻辑 — `_unpack_batch`

`_unpack_batch` 根据 `retrieval_mode` 从 collator 输出中提取 query/positive/negative：

### 6.1 Retrieval Mode 映射

| Mode | Query 提取 | Positive 提取 |
|------|-----------|-------------|
| `t2i` (text→image) | `input_ids`, `attention_mask` | `pixel_values`, `image_grid_thw`, `input_ids`* |
| `i2i` (image→image) | `pixel_values`, `image_grid_thw` | `pixel_values`, `image_grid_thw` |
| `t2t` (text→text) | `input_ids`, `attention_mask` | `pos_input_ids`, `pos_attention_mask` |
| `i2t` (image→text) | `pixel_values`, `image_grid_thw` | `pos_input_ids`, `pos_attention_mask` |
| `m2i` (multi→image) | `input_ids` + `pixel_values` | `pixel_values`, `image_grid_thw` |

> \* VLM 模型（如 Qwen3-VL）的图片项也需要 `input_ids`，因为图片通过 chat template 嵌入到文本序列中（包含 `<|image_pad|>` 占位 token）

### 6.2 Key 优先级

Positive 侧的 key 查找遵循「prefixed key 优先，legacy key 回退」的策略：

```python
# 优先使用 prefixed key
pv = batch.get("pos_pixel_values")
if pv is None:
    # 回退到 legacy key（向后兼容旧 collator）
    pv = batch.get("pixel_values")
```

**注意**：不能使用 `or` 连接 Tensor 值（如 `a or b`），因为多元素 Tensor 的 `__bool__()` 会抛出 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。必须用 `is None` 判断。

---

## 7. 混合精度支持

| 精度模式 | 处理方式 |
|---------|---------|
| fp16 | 创建 `GradScaler`，前向用 `autocast()`，反向用 `scaler.scale(loss).backward()` |
| bf16 | 由 Accelerate 自动处理 autocast，不需要 scaler |
| fp32 | 无特殊处理 |

```python
# 在 GradientCache.step() 中
if self.accelerator:
    mixed = self.accelerator.mixed_precision
    if mixed == "fp16":
        fp16 = True
        scaler = getattr(self.accelerator, "scaler", None)
    # bf16 autocast is handled by Accelerate — no scaler needed
```

---

## 8. DDP 同步优化

多 GPU 训练时，`no_sync_except_last=True` 让前 N-1 个 chunk 跳过 DDP 的 all-reduce 同步，只在最后一个 chunk 同步梯度。这显著减少通信开销：

```python
# forward_backward() 中
if no_sync_except_last:
    sync_contexts = [model.no_sync] * (N-1) + [nullcontext]
```

**前提**：所有模型必须是 `DistributedDataParallel` 包装的。

---

## 9. 不同模型类型的兼容性总结

### 9.1 CLIP / SigLIP（双塔模型）

```yaml
encoder_mode: clip_like
```

- **特点**：图片和文本由独立 encoder 处理
- **pixel_values**: `[B, C, H, W]`，dim=0 = batch_size
- **GradCache 兼容性**：标准 split 即可，无需特殊处理
- **Collator**: `default` (VisualRetrievalCollator)

### 9.2 Qwen3-VL（VLM 单塔模型）

```yaml
encoder_mode: qwen3_vl
```

- **特点**：图片通过 chat template 嵌入文本序列，由同一个模型处理
- **pixel_values**: `[total_patches, C, patch_H, patch_W]`，dim=0 = 总 patch 数（≠ batch_size）
- **额外依赖**: `qwen_vl_utils`（用于 `process_vision_info`）
- **GradCache 兼容性**：需要 `_split_vlm_inputs` 自定义分割
- **Collator**: `qwen3_vl` (QwenVisualRetrievalCollator)
- **特殊处理**:
  - 纯文本 query 不调用 `process_vision_info`，不生成 `pixel_values`
  - 图片项同时需要 `input_ids`（包含图片占位 token）
  - `pixel_values` 和 `image_grid_thw` 必须配对正确分割

### 9.3 Composed 模式（文本+图片独立模型）

```yaml
encoder_mode: composed
```

- **特点**：文本和图片分别用不同模型编码
- **GradCache 兼容性**：标准 split，各 encoder 独立
- **Collator**: `composed`

---

## 10. 注意事项与常见陷阱

### 10.1 Tensor 布尔值陷阱

```python
# WRONG: Tensor 不能用 or 判断
pv = batch.get("pos_pixel_values") or batch.get("pixel_values")
# RuntimeError: Boolean value of Tensor with more than one value is ambiguous

# CORRECT: 用 is None 判断
pv = batch.get("pos_pixel_values")
if pv is None:
    pv = batch.get("pixel_values")
```

### 10.2 VLM pixel_values 分割

```python
# WRONG: 按 batch_size 在 dim=0 分割（CLIP 可以，VLM 不行）
pixel_values.split(chunk_size, dim=0)  # dim=0 是 total_patches，不是 B

# CORRECT: 根据 image_grid_thw 计算每张图的 patch 数
patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
```

### 10.3 VLM 图片项需要 input_ids

Qwen3-VL 等 VLM 不同于 CLIP：即使是「图片」项，模型也需要 `input_ids`（chat template 生成的文本序列中包含图片占位 token）。忘记传 `input_ids` 会报：

```
ValueError: You must specify exactly one of input_ids or inputs_embeds
```

### 10.4 Collator 输出 key 命名规范

```
query 侧:  query_input_ids, query_attention_mask, query_pixel_values, query_image_grid_thw
positive 侧: pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw
negative 侧: neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw
legacy 别名: input_ids (= query), attention_mask (= query), pixel_values (= positive)
```

### 10.5 encoder_mode 必须正确传递

`cli.py` 生成的训练 config YAML 必须包含 `encoder_mode`，否则 `train.py` 会回退到 `"auto"`，导致使用错误的 collator。

### 10.6 Processor 加载

Qwen3-VL 不能用 `AutoProcessor.from_pretrained()` 加载（会触发 `transformers` 视频处理模块的 bug），必须用 `Qwen3VLProcessor.from_pretrained()`。这已在 `ProcessorRegistry` 中通过 `qwen3_vl` loader 处理。

---

## 11. 扩展指南

### 11.1 添加新 VLM 模型支持

如果新 VLM 的 `pixel_values` 也不是 `[B, ...]` 格式，需要：

1. **Processor**: 在 `vembed/model/processors/` 下注册新的 `ProcessorLoader`
2. **Collator**: 在 `vembed/data/collators/` 下注册新的 Collator（参考 `qwen.py`）
3. **Backbone**: 在 `vembed/model/backbones/` 下注册新的模型
4. **Split 函数**: 如果 `pixel_values` 的分割逻辑不同，需要修改 `_split_vlm_inputs`

当前 `_split_vlm_inputs` 使用 `image_grid_thw` 来计算 patch 数（`t * h * w`），适用于 Qwen-VL 系列。如果新模型的 patch 计算方式不同，需要扩展该函数。

### 11.2 自定义 Loss 函数

Loss 函数签名必须是 `loss_fn(query_reps, positive_reps, [negative_reps])`:

```python
def my_loss(q_emb: Tensor, p_emb: Tensor, n_emb: Tensor | None = None) -> Tensor:
    ...
    return scalar_loss
```

GradCache 会将所有 model 的 concatenated representations 按顺序传给 `loss_fn`。

---

## 12. API 参考

::: vembed.training.gradient_cache

::: vembed.grad_cache
