# 知识蒸馏实战：从大模型到小模型的精度迁移

当你有一个高精度但 inference 缓慢的大模型，想要快速且轻量级的部署方案时，**知识蒸馏 (Knowledge Distillation)** 是最有效的技术。本教程展示如何用大模型（Teacher）训练小模型（Student），实现 95%+ 的精度保留，同时显著加快推理。

---

## 1. 为什么需要知识蒸馏？

```
大模型 (Teacher)          小模型 (Student)
├─ Qwen3-VL-8B           ├─ Qwen3-VL-2B
├─ 参数：7.6B            ├─ 参数：2.7B
├─ 精度：78% R@1          ├─ 精度：74% R@1（无蒸馏）
├─ 推理：500ms            ├─ 推理：200ms
└─ 成本：$$$              └─ 成本：$

知识蒸馏目标：
  Student 精度：74% → 77%（接近 Teacher）
  推理时间：500ms → 200ms（保持快速）
  参数量：保持 2.7B（轻量）
```

### 性能对比

| 方案 | 精度 (Recall@1) | 参数量 | 推理速度 | 存储 |
|------|----------------|--------|--------|------|
| **Teacher (Qwen3-VL-8B)** | 78% | 7.6B | 慢 | 30GB |
| **Student (无蒸馏)** | 74% | 2.7B | 快 | 10GB |
| **Student (蒸馏)** | **77%** | 2.7B | 快 | 10GB |

**关键优势：** 仅用 2.7B 模型达到接近 8B 的精度！

---

## 2. 知识蒸馏原理

### 2.1 三个损失函数

```
总损失 = α × CE_Loss + (1-α) × KD_Loss

1. 标准交叉熵损失 (CE Loss)：
   L_CE = -Σ log(P_student(y))

   作用：Student 学习真实标签

2. 知识蒸馏损失 (KD Loss)：
   L_KD = KL(P_teacher, P_student)

   作用：Student 学习 Teacher 的预测分布

3. 总体效果：
   Student = 既学真实标签，又学 Teacher 的泛化特征
```

### 2.2 温度参数 (Temperature)

```
温度 T = 1（默认）：
  Teacher: 78%  Student: 77%（严格）

温度 T = 3（推荐）：
  Teacher: 75%  Student: 76%（软化，更容易学）

温度 T = 5（极端）：
  Teacher: 70%  Student: 74%（过于软化，信息丢失）

建议：T = 3-4
```

---

## 3. 数据与配置

### 3.1 训练数据准备

```python
# 相同的数据集用于 Teacher 和 Student
data = load_data("data/flickr30k/train.jsonl")

# 使用相同的 validation set 评测
# 但注意：Student 已看过这些数据（来自 CE Loss）
# 需要单独的 test set 进行最终评测
```

### 3.2 蒸馏配置

```yaml
# distillation_config.yaml
# ========== Teacher 模型 ==========
teacher_model_name_or_path: "experiments/qwen3_8b_finetuned/checkpoint-final"

# ========== Student 模型 ==========
model_name_or_path: "Qwen/Qwen3-VL-Embedding-2B"

# ========== 蒸馏参数 ==========
use_distillation: true
distillation_loss_type: "kl_divergence"    # KL 散度
temperature: 4.0                           # 温度参数
distillation_weight: 0.5                   # 蒸馏损失权重

# ========== 基础训练参数 ==========
data_path: "data/flickr30k/train.jsonl"
val_data_path: "data/flickr30k/val.jsonl"

output_dir: "experiments/student_distilled"
epochs: 5                          # 蒸馏需要更多 epoch
batch_size: 64
learning_rate: 1.5e-5

# ========== 内存优化 ==========
use_lora: true
lora_r: 16
use_gradient_cache: true

# ========== 日志 ==========
logging_steps: 10
save_steps: 0
eval_strategy: "epoch"
```

### 3.3 启动蒸馏训练

```bash
# 单 GPU
python run.py distillation_config.yaml

# 多 GPU
accelerate launch run.py distillation_config.yaml

# CLI 覆盖
python run.py distillation_config.yaml \
    --config_override \
        teacher_model_name_or_path="path/to/teacher" \
        temperature=4.0 \
        distillation_weight=0.5 \
        epochs=5
```

---

## 4. 性能对比与评测

### 4.1 完整对比实验

```python
from vembed import Predictor
import numpy as np

# 加载三个模型
teacher = Predictor("experiments/qwen3_8b_finetuned")
student_raw = Predictor("experiments/qwen3_2b_raw")
student_distilled = Predictor("experiments/student_distilled/checkpoint-final")

# 编码测试数据
queries = [...]  # 1000 个测试查询
images = [...]   # 100k 个候选图片

# Teacher 编码
t_query_embs = teacher.encode_text(queries)
t_image_embs = teacher.encode_image(images)

# Student (无蒸馏)
s_raw_query_embs = student_raw.encode_text(queries)
s_raw_image_embs = student_raw.encode_image(images)

# Student (蒸馏)
s_dist_query_embs = student_distilled.encode_text(queries)
s_dist_image_embs = student_distilled.encode_image(images)

# 评测
def compute_recall_at_1(query_embs, image_embs):
    sims = np.dot(query_embs, image_embs.T)
    return (np.argmax(sims, axis=1) == np.arange(len(query_embs))).mean()

teacher_r1 = compute_recall_at_1(t_query_embs, t_image_embs)
student_raw_r1 = compute_recall_at_1(s_raw_query_embs, s_raw_image_embs)
student_dist_r1 = compute_recall_at_1(s_dist_query_embs, s_dist_image_embs)

print(f"Teacher:              {teacher_r1:.2%}")
print(f"Student (无蒸馏):      {student_raw_r1:.2%}")
print(f"Student (蒸馏):        {student_dist_r1:.2%}")
print(f"精度保留率:           {(student_dist_r1 / teacher_r1) * 100:.1f}%")
```

### 4.2 预期结果

```
基准数据（Flickr30k，5 epoch 蒸馏）：

Teacher (Qwen3-VL-8B)：        78.2%
Student (Qwen3-VL-2B, raw)：   74.3%
Student (Qwen3-VL-2B, 蒸馏)：  77.1%

精度保留率：77.1% / 78.2% = 98.6% ✓
精度提升：77.1% - 74.3% = +2.8 pp
推理速度提升：8B → 2B = 4x 快速
```

---

## 5. 实战优化技巧

### 5.1 三阶段蒸馏

```yaml
# 阶段 1：低温蒸馏（严格学习）
epochs: 2
temperature: 2.0
distillation_weight: 0.7

# 阶段 2：中温蒸馏（平衡学习）
epochs: 2
temperature: 4.0
distillation_weight: 0.5

# 阶段 3：高温蒸馏（泛化学习）
epochs: 1
temperature: 6.0
distillation_weight: 0.3
```

### 5.2 混合蒸馏

```python
# 结合多个 Teacher
teachers = [
    Predictor("teacher_qwen3_8b"),
    Predictor("teacher_clip_large"),
]

# 集成 Teacher 的预测
teacher_predictions = []
for teacher in teachers:
    pred = teacher.encode_text(queries)
    teacher_predictions.append(pred)

# 平均 Teacher 的预测（集成）
teacher_ensemble_pred = np.mean(teacher_predictions, axis=0)

# Student 学习集成预测
# loss = KL(teacher_ensemble_pred, student_pred)
```

### 5.3 在线蒸馏

```yaml
# Teacher 和 Student 都在线训练（非固定 Teacher）
update_teacher_every: 100  # 每 100 步更新 Teacher
teacher_momentum: 0.999    # Teacher 使用 EMA 更新

# 优势：Teacher 和 Student 共同进步
```

---

## 6. 推理部署

### 6.1 模型选择

```python
# 原始 Teacher（精度最高，但慢）
# → 用于离线批处理、生成初始排序

# Student + 蒸馏（精度次高，快速）
# → 用于实时 API 服务

# 两阶段检索
# 第 1 阶段：快速检索（Student）→ 候选 1000
# 第 2 阶段：精确重排（Teacher）→ 最终 Top 10

# 性能：准确性 = 98% × Teacher，速度 = 10 × Student
```

### 6.2 量化 + 蒸馏（极限优化）

```bash
# 先蒸馏到 Student，再量化
python distillation_training.py

# 量化 Student
# INT8 量化：存储 2.7GB 显存 → 0.7GB
python quantize_model.py experiments/student_distilled/
```

---

## 7. 常见问题

### Q1：蒸馏需要多少数据？

**A：** 与原始微调相同

```
推荐：
  < 1k 数据：1-2 epoch
  1k-10k：3-5 epoch
  > 10k：5-10 epoch

关键：足量数据避免过拟合
```

### Q2：温度参数如何选择？

**A：**

```
T = 1-2：严格，快速收敛，但可能欠学
T = 3-4：平衡（推荐）
T = 5+：宽松，学习缓慢，需要更多 epoch

经验：
  开始用 T=3-4
  如果收敛慢 → 增加 T
  如果精度不够 → 减小 T
```

### Q3：蒸馏 vs LoRA，选哪个？

**A：**

| 方案 | 精度 | 显存 | 推理速度 | 场景 |
|------|------|------|--------|------|
| **LoRA** | 96% | 低 | 同原模型 | 轻量微调 |
| **蒸馏** | 98% | 中 | 4x 快速 | 模型压缩 |

选择：
- 如果只要轻量微调 → LoRA
- 如果要快速推理 → 蒸馏
- 如果都要 → 先蒸馏再 LoRA

---

## 8. 总结

**知识蒸馏的完整流程：**

```
Step 1: 训练高精度 Teacher（6-12 小时）
        ├─ 微调大模型
        └─ 验证精度 > 75%

Step 2: 蒸馏训练 Student（2-4 小时）
        ├─ 同样数据集
        ├─ 使用 KL 散度 + CE Loss
        └─ 5 epoch

Step 3: 评测与对比
        ├─ Student 精度：98% Teacher
        ├─ 推理速度：4x 快速
        └─ 参数量：1/3 Student

Step 4: 部署
        ├─ 单个 Student 模型
        └─ 完整生产系统
```

**关键成果：**
- ✅ 仅用 2.7B 模型达到接近 8B 的精度
- ✅ 推理速度提升 4 倍
- ✅ 模型大小减小 3 倍

---

**相关教程：**
- [Qwen3-VL 微调](./02_qwen3_multimodal_retrieval_zh.md)
- [参数高效微调](./04_parameter_efficient_tuning_zh.md)

