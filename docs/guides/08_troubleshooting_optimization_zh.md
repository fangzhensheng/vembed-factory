# 问题诊断与性能优化指南：常见 Bug 的快速解决方案

本教程是一份 **问题解决手册**，涵盖训练、推理、部署中的 50+ 个常见问题及其解决方案。

---

## 1. 显存问题诊断

### 症状：CUDA Out of Memory

**原因排查：**

```python
# 诊断脚本
import torch

def diagnose_gpu():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"已占用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    print(f"缓存: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
    print(f"最大占用: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

diagnose_gpu()
```

**解决方案（按优先级）：**

```yaml
# 步骤 1：降低 batch_size
batch_size: 128 → 64 → 32 → 16 → 8

# 步骤 2：启用梯度缓存
use_gradient_cache: true
gradient_cache_chunk_size: 64 → 32 → 16 → 8

# 步骤 3：启用 LoRA
use_lora: true
lora_r: 16 → 8 → 4

# 步骤 4：梯度累积替代大 batch
batch_size: 32
gradient_accumulation_steps: 2  # 实际 batch = 64

# 步骤 5：使用更小模型
model: CLIP-B → CLIP-S
# 或
model: Qwen3-VL-2B → 使用量化版本

# 步骤 6：启用模型量化
torch_dtype: "float32" → "float16" / "bfloat16"
```

**快速修复脚本：**

```bash
# 自动尝试配置（从严格到宽松）
for batch in 128 64 32 16 8; do
    for use_gc in true false; do
        python run.py config.yaml \
            --config_override batch_size=$batch use_gradient_cache=$use_gc
        echo "尝试: batch=$batch, grad_cache=$use_gc"
    done
done
```

---

## 2. 训练不收敛

### 症状：Loss 不下降或波动剧烈

**诊断：**

```python
# 查看训练日志
import json

logs = json.load(open("training_logs.json"))
losses = [l['loss'] for l in logs]

# 检查波动
import numpy as np
volatility = np.std(losses[-100:]) / np.mean(losses[-100:])

if volatility > 0.2:
    print("⚠️ Loss 波动剧烈！")
```

**解决方案：**

| 症状 | 原因 | 解决 |
|------|------|------|
| Loss 随机波动 | 学习率太高 | 降低 LR（2e-5 → 1e-5） |
| Loss 单调平坦 | 学习率太低 | 提高 LR（1e-5 → 2e-5） |
| Loss 无法收敛 | 数据问题 | 检查数据格式、正负样本 |
| Loss 先降后升 | 过拟合 | 增加 epochs / 早停 |
| Loss 爆炸 | 梯度爆炸 | 增加 max_grad_norm（1.0 → 2.0） |

**快速检查清单：**

```yaml
检查项:
  □ 学习率是否在 1e-5 ~ 5e-5
  □ Batch size 是否足够大（>= 32）
  □ 数据格式是否正确（JSONL 有效）
  □ 正样本和负样本是否正确配对
  □ 温度参数是否合理（0.05 ~ 0.1）
  □ warmup_steps 是否足够（总步数 10%）
```

---

## 3. 推理质量问题

### 症状：Recall@1 很低（< 50%）

**诊断步骤：**

```python
# 1. 检查模型是否加载正确
predictor = Predictor("checkpoint")
print(f"模型权重数: {sum(p.numel() for p in model.parameters())}")

# 2. 检查 embedding 是否正常
test_text = "test query"
emb = predictor.encode_text(test_text)
print(f"Embedding 范围: [{emb.min():.2f}, {emb.max():.2f}]")
print(f"Embedding norm: {np.linalg.norm(emb):.2f}")

# 3. 检查相似度分布
sims = []
for i in range(len(queries)):
    for j in range(len(images)):
        sim = np.dot(query_embs[i], image_embs[j])
        sims.append(sim)

print(f"相似度范围: {np.min(sims):.2f} ~ {np.max(sims):.2f}")
print(f"相似度均值: {np.mean(sims):.2f}")
```

**常见原因和解决方案：**

```
原因 1：模型没有正确微调
  症状：推理时 Recall 与 Zero-shot 基本相同
  解决：
    ✓ 检查 checkpoint 是否正确
    ✓ 检查微调时是否启用了 LoRA
    ✓ 验证 loss 在训练期间是否下降

原因 2：评测数据与训练数据分布不同
  症状：训练集 Recall 高，测试集低
  解决：
    ✓ 检查训练集和测试集的相似性
    ✓ 进行 Domain Adaptation 微调
    ✓ 增加训练数据的多样性

原因 3：向量库距离度量错误
  症状：手动计算相似度高，但索引搜索低
  解决：
    ✓ 检查 FAISS 距离类型（L2 vs 余弦）
    ✓ 确保 embedding 已归一化
    ✓ 验证索引是否正确构建
```

---

## 4. 数据处理问题

### 症状：数据加载失败或格式错误

**检查脚本：**

```python
import json

# 验证 JSONL 格式
def validate_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                assert 'query' in record, "缺少 query 字段"
                assert 'positive' in record, "缺少 positive 字段"
            except Exception as e:
                print(f"第 {line_no} 行错误: {e}")
                return False
    return True

validate_jsonl("data/train.jsonl")
```

**常见错误和修复：**

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `KeyError: 'positive'` | JSONL 缺少字段 | 重新生成 JSONL |
| `FileNotFoundError` | 图片路径不存在 | 检查 image_root 路径 |
| `UnicodeDecodeError` | 编码问题 | 确保 UTF-8 编码 |
| `JSON decode error` | 格式错误 | 逐行检查 JSONL |

---

## 5. 分布式训练问题

### 症状：多卡训练时精度不一致或 loss 不同步

**诊断：**

```bash
# 检查 GPU 通信
NCCL_DEBUG=TRACE accelerate launch run.py config.yaml 2>&1 | grep TRACE

# 检查显存分配
nvidia-smi -l 1

# 验证所有进程正常
ps aux | grep python
```

**常见问题和解决方案：**

```yaml
问题 1: "RuntimeError: expected scalar type Half but found Float"
  原因：数据类型不匹配（float16 vs float32）
  解决：
    ✓ 统一 torch_dtype
    ✓ 检查输入数据类型

问题 2: "NCCL operation timed out"
  原因：网络连接缓慢或节点间通信问题
  解决：
    ✓ 减少梯度同步频率（logging_steps）
    ✓ 使用 gradient_accumulation 减少通信
    ✓ 检查网络带宽

问题 3: Loss 在不同进程上不一致
  原因：随机种子不同或 BN 统计不同步
  解决：
    ✓ 设置相同的随机种子
    ✓ 启用 sync_bn
```

---

## 6. 推理性能优化

### 目标：加快推理速度

**优化方案（按难度）：**

```
Level 1：简单优化（5 分钟）
  □ 批量推理（batch_size > 1）
  □ GPU 预热（推理前 warm up）
  □ 关闭不必要的计算图（torch.no_grad()）

Level 2：中等优化（30 分钟）
  □ 模型量化（FP16 / INT8）
  □ 启用 TorchScript 编译
  □ 使用更小的模型（LoRA）

Level 3：高级优化（数小时）
  □ ONNX 转换
  □ TensorRT 编译
  □ 知识蒸馏到更小模型
  □ 模型剪枝
```

**实际优化：**

```python
import torch
from torch.quantization import quantize_dynamic

# 量化模型
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 测试推理速度
import time
start = time.time()
for _ in range(100):
    _ = quantized_model(input_ids)
elapsed = time.time() - start
print(f"推理速度: {100 / elapsed:.1f} samples/sec")
```

**性能基准：**

```
模型：Qwen3-VL-2B
设备：A100 GPU

默认推理：20ms per sample = 50 samples/sec
FP16 量化：15ms per sample = 67 samples/sec（+34%）
INT8 量化：12ms per sample = 83 samples/sec（+66%）
TensorRT：8ms per sample = 125 samples/sec（+150%）
```

---

## 7. 部署问题

### 症状：API 服务运行缓慢或经常 timeout

**诊断和优化：**

```python
# 监控 API 延迟
@app.post("/search")
async def search(query: str):
    start = time.time()

    # Step 1: 编码查询
    query_time = time.time()
    query_emb = predictor.encode_text(query)
    encode_time = time.time() - query_time

    # Step 2: FAISS 搜索
    search_time = time.time()
    distances, indices = index.search(query_emb.reshape(1, -1), k=10)
    search_time = time.time() - search_time

    # Step 3: 后处理
    post_time = time.time()
    results = format_results(indices[0])
    post_time = time.time() - post_time

    total_time = time.time() - start

    print(f"编码: {encode_time*1000:.1f}ms, FAISS: {search_time*1000:.1f}ms, 后处理: {post_time*1000:.1f}ms, 总计: {total_time*1000:.1f}ms")

    return results
```

**常见瓶颈和解决方案：**

| 瓶颈 | 典型占比 | 解决方案 |
|------|--------|--------|
| 模型编码 | 50-70% | 使用 GPU，批量处理 |
| FAISS 搜索 | 20-30% | 索引优化（IVF vs FLAT） |
| 后处理 | 5-10% | 缓存元数据，异步处理 |
| 网络 I/O | 5-15% | 连接池，分页返回 |

---

## 8. 快速诊断工具

**一键诊断脚本：**

```bash
#!/bin/bash
# diagnose.sh

echo "=== 环境诊断 ==="
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"

echo "=== 数据诊断 ==="
python -c "
import json
with open('data/train.jsonl') as f:
    sample = json.loads(f.readline())
    print(f'JSONL 样本: {sample}')
"

echo "=== 模型诊断 ==="
python -c "
from vembed import Trainer
trainer = Trainer('openai/clip-vit-base-patch32')
print('✓ 模型加载成功')
"

echo "=== 推理诊断 ==="
python -c "
from vembed import Predictor
predictor = Predictor('checkpoint')
emb = predictor.encode_text('test')
print(f'✓ 推理成功，embedding 维度: {emb.shape}')
"
```

---

## 9. 最佳实践检查清单

**部署前必检：**

```yaml
数据检查：
  □ JSONL 格式正确
  □ 图片路径有效
  □ 没有重复样本
  □ 数据分布合理

训练检查：
  □ Loss 正常下降
  □ Validation 指标在合理范围
  □ 没有过拟合信号
  □ 模型权重已保存

推理检查：
  □ Embedding 维度正确
  □ 相似度范围合理 (-1~1 或 0~1)
  □ 推理速度满足 SLA
  □ 内存占用稳定

部署检查：
  □ GPU 显存充足
  □ 网络连接正常
  □ 监控告警已配置
  □ 容灾备份已准备
```

---

## 10. 求助指南

**遇到问题时的处理流程：**

```
1. 确认问题症状
   ├─ 显存溢出？→ 参考第 1 章
   ├─ Loss 不收敛？→ 参考第 2 章
   ├─ 精度低？→ 参考第 3 章
   └─ 其他？→ 继续下一步

2. 查看日志
   python run.py config.yaml 2>&1 | tee train.log
   tail -100 train.log

3. 运行诊断脚本
   bash diagnose.sh

4. 尝试简化配置
   batch_size 64 → 32
   epochs 5 → 2
   model 大 → 小

5. 查阅项目文档
   GitHub Issues
   官方文档
   社区论坛

6. 提交 issue（带上）
   ✓ 完整错误日志
   ✓ 配置文件
   ✓ 数据样本
   ✓ 系统信息
```

---

## 总结

**快速速查表：**

| 问题 | 症状 | 快速修复 |
|------|------|--------|
| 显存溢出 | CUDA OOM | ↓ batch_size, ✓ grad_cache |
| Loss 不降 | 平坦 loss | ↑ learning_rate |
| Loss 爆炸 | NaN loss | ↑ max_grad_norm |
| 推理慢 | > 1 sec | 批量处理，量化 |
| 精度低 | Recall < 50% | 检查 checkpoint，重新微调 |
| 数据错 | FileNotFoundError | 检查路径，重新生成 JSONL |

---

**相关教程：**
- [CLIP 微调](./01_clip_text_to_image_zh.md)
- [参数高效微调](./04_parameter_efficient_tuning_zh.md)
- [分布式训练](./05_distributed_training_zh.md)

