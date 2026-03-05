# 困难样本挖掘 (Hard Negative Mining)

## 简单高效方案：In-Batch Hard Mining

从当前 batch 中选取 top-K 最困难的负样本，无需额外状态管理。

### 特点
- **简单**: 无需维护 memory bank 或 momentum encoder
- **高效**: 与 Gradient Cache 和 DDP 完全兼容
- **有效**: 对困难样本更专注，收敛更快

### 适用场景
```yaml
# 大 batch size 训练 (推荐)
batch_size: 128-256

# 结合 Gradient Cache 实现超大 batch
use_gradient_cache: true
gradient_cache_chunk_size: 2
```

### 配置

```yaml
loss_type: "in_batch_hard"  # 或 "hard_negative"
temperature: 0.05
hard_topk: 8              # 每个 query 选 8 个最困难负样本
use_all_negatives: false  # false=只 hard negatives, true=所有负样本 (类似 InfoNCE)
```

### 完整示例

```yaml
# Qwen3-VL-Embedding 训练配置
model_name: "Qwen/Qwen3-VL-Embedding-2B"
encoder_mode: "qwen3_vl_embedding"
retrieval_mode: "t2i"

data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

output_dir: "experiments/output_in_batch_hard"
epochs: 3
batch_size: 128
learning_rate: 1.0e-5

# 损失函数
loss_type: "in_batch_hard"
temperature: 0.05
hard_topk: 8

# 内存优化
use_gradient_cache: true
gradient_cache_chunk_size: 2
gradient_checkpointing: true
use_lora: true
```

## API 使用示例

```python
from vembed.losses.functions.hard_negative import InBatchHardMiningLoss

loss_fn = InBatchHardMiningLoss({
    "temperature": 0.05,
    "hard_topk": 8,
    "use_all_negatives": False,
})

loss = loss_fn(query_emb, positive_emb, labels=labels)
```

## 与 InfoNCE 的区别

| 特性 | InfoNCE | In-Batch Hard Mining |
|------|---------|---------------------|
| 负样本 | 全部 in-batch | 仅 top-K 困难样本 |
| 梯度 | 分散到所有负样本 | 集中在困难样本 |
| 收敛 | 较慢 | 较快 |
| 配置 | `loss_type: "infonce"` | `loss_type: "in_batch_hard"` |

**选择建议**:
- 如果 batch size 较小 (≤64): 使用 InfoNCE
- 如果 batch size 较大 (≥128): 使用 In-Batch Hard Mining
- 如果想更快收敛: 使用 In-Batch Hard Mining

## 配置文件

| 配置文件 | 说明 |
|----------|------|
| `examples/qwen3_vl_embedding_in_batch_hard.yaml` | In-Batch Hard Mining |

## 运行训练

```bash
python -m vembed.entrypoints.train \
    --config examples/qwen3_vl_embedding_in_batch_hard.yaml
```
