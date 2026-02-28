# 多卡分布式训练完全指南：DDP vs FSDP

当模型或数据量增大时，单 GPU 训练变得不可行。本教程介绍两种分布式训练方案的原理、配置和最佳实践。

---

## 1. DDP vs FSDP 对比

| 特性 | DDP | FSDP | 场景 |
|------|-----|------|------|
| **模型大小限制** | 单卡显存 | 分布式显存 | FSDP 支持更大模型 |
| **显存节省** | 0% | 30-70% | 显存紧张用 FSDP |
| **吞吐量** | 高 | 中等 | DDP 更快 |
| **通信开销** | 低 | 中等 | DDP 通信效率高 |
| **学习曲线** | 低 | 高 | DDP 更易上手 |

---

## 2. DDP：数据并行

### 2.1 原理

```
GPU 0: 数据 0-31      GPU 1: 数据 32-63      GPU 2: 数据 64-95
  ↓                      ↓                      ↓
[模型副本]           [模型副本]            [模型副本]
  ↓                      ↓                      ↓
[梯度聚合（AllReduce）]
  ↓                      ↓                      ↓
[更新]               [更新]              [更新]
```

### 2.2 配置与启动

**步骤 1：创建分布式配置**

```bash
# 交互式配置
accelerate config
# 选择 multi_gpu，其他默认

# 或创建 accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
```

**步骤 2：启动训练**

```bash
# 单机多卡 DDP
accelerate launch run.py config.yaml

# 或指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run.py config.yaml

# 自定义进程数
accelerate launch --num_processes 4 run.py config.yaml
```

### 2.3 性能表现

```
显存占用 (4x A100)：
  单卡：24GB × 4 = 96GB（分别存储）
  DDP：24GB × 4 = 96GB（相同）

实际吞吐量：
  单卡：100 samples/sec
  DDP (4 卡)：380 samples/sec（线性 scale 效率 95%）

总训练时间：
  单 GPU：10 小时
  DDP (4 GPU)：2.5 小时 ✓
```

---

## 3. FSDP：完全分片数据并行

### 3.1 原理

```
完全模型（层）分布在 4 个 GPU 上：

Layer 0 ────> GPU 0
Layer 1 ────> GPU 1
Layer 2 ────> GPU 2
Layer 3 ────> GPU 3

训练时：
  前向传播：GPU 0 计算 ──> 通信获取其他层 ──> 继续
  反向传播：类似
```

### 3.2 显存优势

```
大模型（8B 参数）：

DDP：每卡需要 32GB 显存（限制）
FSDP：每卡仅需 8GB 显存（4 卡平摊）

✓ FSDP 能处理 DDP 无法处理的大模型
```

### 3.3 配置

```yaml
# accelerate_fsdp_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 4

fsdp_config:
  min_num_params: 2000000      # 分片的最小参数数
  sharding_strategy: FULL_SHARD # 完全分片
  backward_prefetch: BACKWARD_PRE
  forward_prefetch: True        # 重叠通信和计算
  cpu_offload: False            # 关闭 CPU offload（GPU 充足）
```

### 3.4 启动 FSDP 训练

```bash
accelerate launch --config_file accelerate_fsdp_config.yaml run.py config.yaml
```

---

## 4. 分布式训练最佳实践

### 4.1 设置同步 Batch Norm

```yaml
# 使用 SyncBN 确保所有卡的 BN 统计一致
sync_bn: true

# 配置
sync_bn_group_size: 4    # BN 同步的卡数
```

### 4.2 梯度累积 + 分布式

```yaml
# 逻辑 batch size = real batch size × gradient_accumulation_steps
batch_size: 32
gradient_accumulation_steps: 2     # 逻辑 batch = 64

# 4 卡 DDP 的总 batch = 32 × 2 × 4 = 256
```

### 4.3 学习率线性缩放

```
单 GPU 学习率：2.0e-5
多 GPU 学习率：2.0e-5 × sqrt(num_gpus)

原因：梯度是平均值，需要调整学习率以保持收敛性

配置：
use_linear_scaling: true
base_learning_rate: 2.0e-5
num_gpus: 4  # 自动调整为 2.0e-5 × sqrt(4) = 4.0e-5
```

---

## 5. 多机多卡训练

### 5.1 网络配置

```bash
# 节点 1（主节点）
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=8

accelerate launch run.py config.yaml

# 节点 2
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=4           # 4 个 GPU on node 1，所以从 4 开始
export WORLD_SIZE=8

accelerate launch run.py config.yaml
```

### 5.2 Slurm 集群提交

```bash
#!/bin/bash
# submit_slurm.sh
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4

accelerate config

accelerate launch run.py config.yaml
```

---

## 6. 常见问题与调试

### Q1：DDP 训练时精度不一致

**A：** 检查以下几点：

```python
# 设置随机种子
import torch
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保 deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Q2：多卡训练时 loss 不收敛

**A：** 可能原因和解决方案：

```
1. 梯度爆炸：
   → 增加 max_grad_norm
   → 降低 learning_rate

2. Batch Norm 不同步：
   → use_sync_bn: true

3. AllReduce 通信延迟：
   → 检查网络带宽
   → 减少通信频率（increase logging_steps）
```

### Q3：显存不均衡

**A：** 诊断和优化：

```bash
# 监控显存
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 1

# 如果显存不均衡：
# 1. 检查 batch_size 分布
# 2. 检查 gradient_accumulation_steps
# 3. 使用 FSDP 而不是 DDP
```

---

## 7. 性能基准

### 7.1 吞吐量对比

```
模型：Qwen3-VL-2B
数据集：Flickr30k（128 Batch size）

单 GPU (A100)：
  吞吐量：100 samples/sec
  显存：12GB

DDP (4x A100)：
  吞吐量：380 samples/sec（Scale eff: 95%）
  显存：12GB/卡

DDP (8x A100)：
  吞吐量：720 samples/sec（Scale eff: 90%）
  显存：12GB/卡

FSDP (4x A100)：
  吞吐量：350 samples/sec
  显存：6GB/卡
```

### 7.2 训练时间对比

```
训练 3 epoch，50k 样本：

单 GPU：~7 小时
DDP (4 GPU)：~2 小时 ✓
DDP (8 GPU)：~1 小时 ✓✓
FSDP (8 GPU)：~1.2 小时
```

---

## 8. 故障恢复与检查点

### 8.1 断点续训

```yaml
# 配置自动保存
resume_from_checkpoint: "checkpoint-latest"
save_strategy: "steps"
save_steps: 100
```

### 8.2 分布式环境下恢复

```python
# 分布式训练中，仅主进程保存
if rank == 0:
    model.save_pretrained("checkpoint")

# 所有进程同步
dist.barrier()

# 所有进程加载
model.load_pretrained("checkpoint")
```

---

## 9. 推荐配置

### 方案 A：4x GPU（推荐）

```yaml
distributed_type: MULTI_GPU
num_processes: 4

# 训练
batch_size: 128      # 单卡 32 × 4
learning_rate: 2.0e-5 × sqrt(4) = 4.0e-5
gradient_accumulation_steps: 1
```

### 方案 B：8x GPU + 显存紧张

```yaml
distributed_type: FSDP

# FSDP 配置
fsdp_config:
  sharding_strategy: FULL_SHARD
  backward_prefetch: BACKWARD_PRE
  forward_prefetch: True

batch_size: 64       # 单卡
learning_rate: 2.0e-5 × sqrt(8) = 5.66e-5
```

---

## 10. 总结

**选择决策：**

```
GPU 数量 ≤ 4 且模型 ≤ 8B：DDP
GPU 数量 > 4 或模型 > 8B：FSDP
多机多卡：均可，DDP 通常更高效
```

**性能期望：**
- DDP 线性 scale（N 卡 ≈ 1/N 的训练时间）
- FSDP 次线性 scale（通信开销更大）

---

**相关教程：**
- [参数高效微调](./04_parameter_efficient_tuning_zh.md)
- [问题诊断](./troubleshooting_optimization_zh.md)

