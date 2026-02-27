# DINOv2 Embedding Fine-tuning in Practice: Building a High-Precision Product Image Search System

**Image Retrieval (Image Search)** is a core technology in computer vision, widely used in e-commerce product search, visual search engines, and other scenarios. Its core principle involves converting images into vectors (Embeddings) and retrieving relevant images by calculating vector similarity.

**DINOv2**, introduced by Meta, is currently a leading self-supervised vision foundation model. The features it generates possess strong semantic representation capabilities. However, general-purpose pre-trained models often underperform in specific vertical domains (such as fine-grained product retrieval). To distinguish similar but different products (e.g., chairs of different styles), we need to **Fine-tune** the model.

This guide will introduce how to use the open-source tool **[vembed-factory](https://github.com/fangzhensheng/vembed-factory)** to fine-tune the DINOv2 model on the Stanford Online Products (SOP) dataset, significantly improving its retrieval precision.

---

## 1. Why Fine-tuning?

Although DINOv2 has strong generalization capabilities, it is pre-trained on massive general data. When facing fine-grained classification tasks in specific domains (e.g., distinguishing two products that look extremely similar but have different models), using pre-trained features directly (Zero-shot) often fails to meet industrial precision requirements.

The core goal of fine-tuning is to optimize the Embedding space through **Supervised Contrastive Learning**:
- **Pull closer** the embedding distances of images of the same product from different angles (Positive pairs).
- **Push apart** the embedding distances of images of different products (Negative pairs).

Experiments show that after fine-tuning, the model's Recall@1 (top-1 hit rate) on the SOP dataset can improve from about 50% to over 80%.

## 2. Environment Preparation

We will use the `vembed-factory` framework, which encapsulates underlying details such as data loading, Loss calculation, and multi-GPU training, allowing users to focus on model strategies.

It is recommended to use `uv` for environment management to get a faster dependency installation experience:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the project
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# 3. Initialize the environment
uv sync
source .venv/bin/activate
```

## 3. Dataset Preparation: SOP

This tutorial uses the **Stanford Online Products (SOP)** dataset, which contains 120k images covering 22k product categories.

### 3.1 Download Data
Please download the dataset from the [SOP official website](https://cvgl.stanford.edu/projects/lifted_struct/) or Kaggle, and extract it to the `data/stanford_online_products/` directory:

```
data/stanford_online_products/
├── Ebay_train.txt      # Training set index
├── Ebay_test.txt       # Test set index
├── bicycle_final/      # Image folder
├── chair_final/
...
```

### 3.2 Data Preprocessing

The original data is in txt index format. We need to convert it to the JSONL format supported by the framework. The project provides a unified preprocessing script:

```bash
# Run data preprocessing script (automatically downloads SOP dataset if not present)
python examples/prepare_data.py sop_i2i
```

> **Note**: If the dataset hasn't been downloaded yet, the script will automatically download the SOP dataset from Kaggle via `kagglehub`.

After completion, `train.jsonl` and `val.jsonl` will be generated in the directory, with the following format:

```json
{
  "query_image": "bicycle_final/111.jpg",
  "positive": "bicycle_final/112.jpg",
  "label": 1001
}
```

**Data Format Explanation:**
- `query_image` - Query image path (relative path)
- `positive` - Positive sample image path (another view of the same product)
- `label` - Product ID (used for Supervised Contrastive Loss)

During training, the framework automatically generates **negative samples** from other samples in the same batch with different labels (in-batch negatives), no need to specify in advance.

## 4. Model Fine-tuning

We define training parameters through configuration files without modifying the underlying code.

### 4.1 Configuration Analysis (`examples/dinov2_i2i.yaml`)

```yaml
# --- Model Configuration ---
model_name: facebook/dinov2-base     # Default base; choose small/base/large based on VRAM
retrieval_mode: i2i                  # Image-to-Image retrieval mode
use_lora: true                       # Enable LoRA efficient fine-tuning to reduce VRAM usage

# --- Data Paths ---
data_path: data/stanford_online_products/train.jsonl
val_data_path: data/stanford_online_products/val.jsonl
image_root: data/stanford_online_products
output_dir: experiments/output_sop_dinov2_i2i

# --- Training Parameters ---
epochs: 20
batch_size: 128                      # Larger batch size helps contrastive learning
loss_type: infonce                   # Use InfoNCE Loss with Supervised Contrastive Learning
learning_rate: 0.0001
logging_steps: 10
save_steps: 1000
```

### 4.2 Start Training

After confirming the configuration, execute the following command to start training:

```bash
bash examples/run_dinov2_i2i.sh
```

Or use the Python CLI directly:

```bash
python run.py examples/dinov2_i2i.yaml
```

If you need to override configuration parameters, use the CLI:

```bash
python run.py examples/dinov2_i2i.yaml --config_override epochs=30 batch_size=64
```

During training, model weights and logs will be saved in the `experiments/output_sop_dinov2_i2i` directory.

## 5. Performance Evaluation

After training is complete, we need to evaluate the model's retrieval performance on the SOP test set.

Use the built-in evaluation script:

```bash
# --model_path points to the fine-tuned model checkpoint path
python benchmark/run.py sop \
    --model_path experiments/output_sop_dinov2_i2i/checkpoint-1000 \
    --sop_root data/stanford_online_products \
    --batch_size 128
```

### Expected Results

Based on our experimental results on the **Stanford Online Products (SOP)** dataset, the Embedding fine-tuning brings significant performance improvements:

| Model | Metric | Zero-shot (Un-tuned) | Fine-tuned | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv2-base** | **Recall@1** | 55.03% | **79.97%** | **+24.94%** |
| *(facebook/dinov2-base)* | Recall@10 | 71.72% | **91.31%** | **+19.60%** |
| | Recall@100 | 84.70% | **96.37%** | **+11.67%** |

**Training Configuration:**
- Model: facebook/dinov2-base (default)
- Dataset: SOP training set (~120k images)
- Training epochs: 2
- Batch Size: 128
- LoRA: Enabled

After fine-tuning, DINOv2-base achieves **79.97%** Recall@1, a **24.94%** improvement over the zero-shot model, making it suitable for production e-commerce product retrieval systems.

## 6. FAQ

### 6.1 How do I adjust training parameters?

**Method 1: Edit YAML Configuration**
```bash
# Modify examples/dinov2_i2i.yaml and run
python run.py examples/dinov2_i2i.yaml
```

**Method 2: CLI Parameter Override**
```bash
# Override parameters without modifying the config file
python run.py examples/dinov2_i2i.yaml \
    --config_override epochs=30 batch_size=64 learning_rate=0.00005
```

### 6.2 How to use other models?

Modify the `model_name` parameter:

```bash
# Use dinov2-small (lower VRAM usage)
python run.py examples/dinov2_i2i.yaml --config_override model_name=facebook/dinov2-small

# Use dinov2-large (best performance)
python run.py examples/dinov2_i2i.yaml --config_override model_name=facebook/dinov2-large
```

**Model Comparison:**

| Model | Parameters | VRAM Usage | Recommended Scenario |
| :--- | :--- | :--- | :--- |
| dinov2-small | ~21M | Low | Edge devices, fast iteration |
| **dinov2-base** (default) | ~86M | Medium | **Best balance of performance and speed** ✓ |
| dinov2-large | ~300M | High | Maximum accuracy required |

Based on README experiments, **dinov2-base** achieves 79.97% Recall@1 after fine-tuning on SOP, making it the best choice for production environments.

### 6.3 How to train on multiple GPUs?

The framework uses `accelerate` to automatically detect and configure distributed training.

**Method 1: Direct Script (Recommended)**
```bash
# Framework automatically detects available GPUs and enables distributed training
bash examples/run_dinov2_i2i.sh
```

**Method 2: Manual accelerate Configuration**
```bash
# Interactive distributed training setup
accelerate config

# Start training (automatically uses multiple GPUs)
accelerate launch run.py examples/dinov2_i2i.yaml
```

> **Note**: In most cases, running the script directly will automatically utilize all GPUs. Manual `accelerate config` is only needed for special hardware setups or specific distributed strategies.

## 7. Summary

Through **vembed-factory**, we have greatly simplified the Embedding fine-tuning process for visual models:

1. **Data Preparation** - `python examples/prepare_data.py sop_i2i`
   - Automatically downloads or processes the SOP dataset
   - Converts to concise JSONL format (query_image, positive, label)

2. **Configuration Management** - YAML config + CLI parameter override
   - Flexibly specify model, loss function, learning rate, etc.
   - No code modification needed to adjust training strategy

3. **Model Training** - One-click launch
   - Support single-GPU / multi-GPU distributed training
   - Support LoRA parameter-efficient fine-tuning
   - Automatic checkpoint saving

4. **Performance Evaluation** - Built-in evaluation script
   - Evaluate Recall@1, Recall@10, and other metrics on SOP test set
   - Typically achieve 20-30% improvement over zero-shot models

**Key Advantages:**
- ✅ Clean data format (no complex hard negatives configuration)
- ✅ Complete Supervised Contrastive Learning support
- ✅ Automatic in-batch negatives generation
- ✅ Support for multiple loss functions and retrieval modes

This makes building production-grade high-precision product image search systems more efficient and convenient.

If you are interested in this project, welcome to visit the GitHub repository and give it a Star: [vembed-factory](https://github.com/fangzhensheng/vembed-factory)
