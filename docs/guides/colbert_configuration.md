# ColBERT Configuration Guide

## What is ColBERT?

**ColBERT (Contextualized Late Interaction)** is a late-interaction retrieval architecture that differs fundamentally from traditional dense retrieval:

### Dense Retrieval vs. ColBERT

```
Dense Retrieval:
  Text/Image → [Single Vector D-dim] → Cosine Similarity

ColBERT:
  Text/Image → [All Token Vectors L×D] → MaxSim Score
```

**Key Insight**: Instead of comparing single global vectors, ColBERT compares all tokens and uses the **mean of max similarities** as the final score.

**Formula**: `Score(Q, D) = mean_q max_d (q_i · d_j)`

---

## ColBERT Configuration: Three Essential Parameters

### 1️⃣ Loss Function

```yaml
loss_type: colbert
```

**What it does:**
- Implements **MaxSim** scoring mechanism for late-interaction retrieval
- Computes similarities at token level: each query token finds its best matching document token
- Takes mean of max similarities across all query tokens

**Why it matters:**
- Enables fine-grained semantic matching without dimension explosion
- Better for specialized retrieval (e-commerce, medical images, etc.)

---

### 2️⃣ Embedding Output Format

```yaml
pooling_method: none
```

**What it does:**
- Returns **all token embeddings** instead of a single global vector
- Output shape: `[B, L, D]` where L = token sequence length
- For vision models: L = number of patches + 1 (CLS token)
- For LLMs: L = output sequence length

**Why it matters:**
- Essential for ColBERT to perform per-token comparisons
- Must be set to `none` when using `loss_type: colbert`
- Other loss functions (InfoNCE, CoSENT) need global pooling (mean/cls)

**When automatic:**
- If `loss_type: colbert` and `pooling_method` is not specified → auto-set to `"none"`
- If `loss_type` is other and `pooling_method="none"` → auto-corrected to `"cls"` with warning

---

### 3️⃣ Token Optimization (Optional)

```yaml
topk_tokens: 32
```

**What it does:**
- Enables attention-guided token pruning
- Keeps only top-K most relevant tokens per sample
- Uses CLS-patch cosine similarity to score token importance

**Why it matters:**
- Reduces memory usage and computation time during inference
- For vision models: ~256 patches → keep only 32 reduces 8× computation
- Does NOT significantly harm accuracy (keeps most relevant patches)

**Common values:**
- `topk_tokens: 0` → Keep all tokens (default, no pruning)
- `topk_tokens: 32` → Recommended for vision models (good balance)
- `topk_tokens: 64` → For higher accuracy with more compute

---

## Complete Configuration Template

```yaml
# ===== ColBERT Core =====
loss_type: colbert              # Required: MaxSim scoring
pooling_method: none            # Required: token-level embeddings

# ===== ColBERT Optional =====
topk_tokens: 32                 # Recommended for vision: attention-guided pruning
projection_dim: 128             # Optional: project embeddings to lower dimension

# ===== Standard Training =====
epochs: 20
batch_size: 128
learning_rate: 5.0e-05
use_gradient_cache: true        # Recommended: 4× batch size on same VRAM
use_lora: true                  # Recommended: 0.1% parameters only

# ===== Data & Output =====
data_path: data/train.jsonl
val_data_path: data/val.jsonl
output_dir: experiments/output_colbert

# ===== Logging =====
logging_steps: 10
save_steps: 500
report_to: none
```

---

## Practical Examples

### DINOv2 + ColBERT (Image-to-Image Retrieval)

```yaml
# DINOv2 ColBERT configuration for fine-grained product search
model_name: facebook/dinov2-base
retrieval_mode: i2i
use_lora: true

# ===== ColBERT Configuration =====
loss_type: colbert              # Late-interaction scoring
pooling_method: none            # [B, 257, 768] token embeddings
topk_tokens: 32                 # DINOv2 has 256 patches + 1 CLS
projection_dim: 128             # Optional: reduces 768 → 128

# ===== Data =====
data_path: data/stanford_online_products/train.jsonl
val_data_path: data/stanford_online_products/val.jsonl
image_root: data/stanford_online_products
output_dir: experiments/output_dinov2_colbert

# ===== Training =====
epochs: 20
batch_size: 128
eval_batch_size: 32
learning_rate: 1.0e-04
use_gradient_cache: true        # Train with 512+ batch on 24GB GPU
```

### Qwen2-7B + ColBERT (Text-to-Text Retrieval)

```yaml
# Qwen2 ColBERT configuration for dense text retrieval
model_name: Qwen/Qwen2-7B-Instruct
retrieval_mode: t2t
use_lora: true

# ===== ColBERT Configuration =====
loss_type: colbert              # Late-interaction scoring
pooling_method: none            # [B, seq_len, 4096] token embeddings
projection_dim: 128             # Reduce 4096 → 128 for efficiency

# ===== Data =====
data_path: data/train.jsonl
val_data_path: data/val.jsonl
output_dir: experiments/output_qwen_colbert

# ===== Training =====
epochs: 3
batch_size: 64
learning_rate: 5.0e-05
use_gradient_cache: true        # Critical for large LLMs
gradient_cache_chunk_size: 4
```

---

## ColBERT vs. Other Retrieval Methods

| Method | Loss | Pooling | Shape | Best For | Memory |
|--------|------|---------|-------|----------|--------|
| **Dense** | InfoNCE | mean/cls | `[B, D]` | Fast retrieval, web-scale | Low |
| **ColBERT** | colbert | **none** | **[B, L, D]** | **Fine-grained matching** | **Higher** |
| **MRL** | mrl | mean/cls | `[B, D]` | Multi-scale retrieval | Low |
| **Distillation** | kl | mean/cls | `[B, D]` | Model compression | Low |

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Wrong Pooling Method

```yaml
# WRONG: This won't work
loss_type: colbert
pooling_method: mean            # ← Error: mean removes token info

# CORRECT: Use "none" for ColBERT
loss_type: colbert
pooling_method: none            # ← Required: preserves all tokens
```

### ❌ Mistake 2: ColBERT with Non-Token Loss

```yaml
# WRONG: Incompatible combination
loss_type: infonce
pooling_method: none            # ← Doesn't make sense with InfoNCE

# CORRECT: Use matching loss + pooling
loss_type: infonce
pooling_method: cls             # ← InfoNCE works with global pooling
```

### ❌ Mistake 3: Omitting pooling_method

```yaml
# WORKS BUT NOT EXPLICIT (not recommended)
loss_type: colbert
# pooling_method not specified → auto-set to "none"

# RECOMMENDED: Always be explicit
loss_type: colbert
pooling_method: none            # Make intent clear
```

---

## Memory Considerations

### Token Embeddings vs. Dense Embeddings

```
Dense Retrieval:
  Batch: [B, D] → [32, 768] = ~24KB per batch
  60K documents: 60K × 768 = ~180MB index

ColBERT with topk_tokens=32:
  Batch: [B, 32, D] → [32, 32, 768] = ~768KB per batch
  60K documents: 60K × 32 × 768 = ~1.4GB index
  (8× larger but enables finer-grained matching)
```

### Optimization Strategies

1. **Use `topk_tokens`** → Reduces token count from 256 to 32 (8× reduction)
2. **Enable `use_gradient_cache`** → Trade compute time for memory (4× batch size)
3. **Use `projection_dim`** → Reduce dimension 768 → 128 (6× reduction)
4. **Enable LoRA** → Only fine-tune 0.1% of parameters

**Example**: DINOv2 ColBERT with all optimizations:
- Base model: 87M parameters
- LoRA: 0.1M parameters (99.9% reduction)
- Tokens: 256 → 32 (via topk_tokens)
- Dimension: 768 → 128 (via projection_dim)
- Batch multiplier: 128 → 512 (via gradient_cache)

---

## Validation Checklist

After setting up ColBERT, verify your configuration:

### Step 1: Check Saved Config
```bash
cat experiments/output_*/.*train_config.yaml | grep -E "loss_type|pooling_method|topk_tokens"
```

Should output:
```yaml
loss_type: colbert
pooling_method: none
topk_tokens: 32
```

### Step 2: Inspect First Batch
```python
import torch
from vembed.data import load_train_data

train_loader = load_train_data("data/train.jsonl", batch_size=2)
batch = next(iter(train_loader))

# Query embeddings shape
query_emb = batch["query_embeddings"]
print(f"Query embeddings: {query_emb.shape}")  # Should be [B, L, D] for ColBERT
# Expected: [2, 257, 768] for DINOv2
```

### Step 3: Run Dry-Run Training
```bash
python run.py examples/dinov2_colbert.yaml --dry_run
# Should complete without OOM errors
```

---

## Troubleshooting

### ⚠️ Issue: OOM During Validation with ColBERT

**Cause**: Computing MaxSim on 60K images creates huge intermediate tensors

**Solution**:
- Set `eval_batch_size` smaller: `eval_batch_size: 16`
- Enable `use_gradient_cache: true` for training
- Increase `topk_tokens` to prune more aggressively

### ⚠️ Issue: Training Is Too Slow

**Cause**: Token-level similarity computation is expensive

**Solution**:
- Enable `topk_tokens: 32` (or higher) for pruning
- Enable `use_gradient_cache: true`
- Use `projection_dim: 128` to reduce dimension
- Increase `batch_size` (gradient cache makes this feasible)

### ⚠️ Issue: Accuracy Didn't Improve Much

**Cause**: ColBERT needs larger batches for contrastive learning

**Solution**:
- Increase `batch_size` (64 → 128 → 256)
- Enable `use_gradient_cache: true` if memory-limited
- Use higher `epochs` (20 → 30+)
- Lower `learning_rate` slightly (1e-4 → 5e-5)

---

## FAQ

**Q: When should I use ColBERT vs. dense retrieval?**
A: Use ColBERT when:
- Fine-grained semantic matching matters (e-commerce, medical)
- You have domain-specific data to fine-tune on
- Accuracy is more important than speed
- You have enough compute/memory for token-level embeddings

Use dense retrieval when:
- Speed is critical (web-scale, real-time)
- Limited compute/memory
- Generic retrieval task

---

**Q: Can I use ColBERT with models other than DINOv2?**
A: Yes! ColBERT works with any vision or text model:
- Vision: DINOv3, ViT, ResNet (with appropriate adapter)
- Text: Qwen, LLaMA, BERT
- Multi-modal: Qwen-VL, LLaVA

---

**Q: What's the difference between `topk_tokens` and `projection_dim`?**
A:
- `topk_tokens`: Selects top-K tokens (e.g., 256 → 32)
- `projection_dim`: Reduces embedding dimension (e.g., 768 → 128)

Both reduce memory/compute, but:
- `topk_tokens` preserves embedding dimension (keeps spatial info)
- `projection_dim` compresses embedding space (loses some info)

Use both together for maximum efficiency!

---

**Q: How do I deploy a fine-tuned ColBERT model?**
A: Same as any model:
```python
from vembed.model import EmbeddingModel

# Load model
model = EmbeddingModel.from_pretrained("experiments/output_colbert/checkpoint-best")

# Forward pass returns [B, L, D] for ColBERT
embeddings = model(images)  # [32, 32, 128] if topk_tokens=32, projection_dim=128

# MaxSim scoring in inference
score = (embeddings[q] @ embeddings[d].T).max(dim=-1).values.mean()
```

---

## Related Resources

- [ColBERT Paper](https://arxiv.org/abs/2004.12832)
- [Framework Architecture](../ARCHITECTURE.md)
- [Training Configuration](./configuration.md)
- [Loss Functions Reference](../../vembed/losses/)
- [Model Backbones](../../vembed/model/backbones/)
