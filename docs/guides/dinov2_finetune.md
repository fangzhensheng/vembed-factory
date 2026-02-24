# Fine-tuning DINOv2: Building a High-Precision "Image Search" Engine

**Image Retrieval (Image Search)** is a core technology in computer vision, widely used in e-commerce product search, visual search engines, and other scenarios. Its core principle involves converting images into vectors (Embeddings) and retrieving relevant images by calculating vector similarity.

**DINOv2**, introduced by Meta, is currently a leading self-supervised vision foundation model. The features it generates possess strong semantic representation capabilities. However, general-purpose pre-trained models often underperform in specific vertical domains (such as fine-grained product retrieval). To distinguish similar but different products (e.g., chairs of different styles), we need to **Fine-tune** the model.

This guide will introduce how to use the open-source tool **[vembed-factory](https://github.com/fangzhensheng/vembed-factory)** to fine-tune the DINOv2 model on the Stanford Online Products (SOP) dataset, significantly improving its retrieval precision.

---

## 1. Why Fine-tuning?

Although DINOv2 has strong generalization capabilities, it is pre-trained on massive general data. When facing fine-grained classification tasks in specific domains (e.g., distinguishing two products that look extremely similar but have different models), using pre-trained features directly (Zero-shot) often fails to meet industrial precision requirements.

The core goal of fine-tuning is to optimize the vector space through **Contrastive Learning**:
- **Pull closer** the vector distances of images of the same product from different angles (Positive pairs).
- **Push apart** the vector distances of images of different products (Negative pairs).

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
The original data is in txt index format. We need to convert it to the JSONL format supported by the framework. The project provides a preprocessing script:

```bash
# Run data preprocessing script
python examples/prepare_sop_data.py --sop_root data/stanford_online_products
```

After completion, `train.jsonl` and `val.jsonl` will be generated in the directory, with the following format:

```json
{"query_image": "bicycle_final/111.jpg", "positive": "bicycle_final/112.jpg", "label": 1001}
```

This format defines positive sample pairs in retrieval tasks (Query Image and Positive Image belong to the same product).

## 4. Model Fine-tuning

We define training parameters through configuration files without modifying the underlying code.

### 4.1 Configuration Analysis (`examples/dinov2_i2i.yaml`)

```yaml
# --- Model Configuration ---
model_type: "custom"
model_name: "facebook/dinov2-small"  # Choose small/base/large based on VRAM
retrieval_mode: "i2i"                # Image-to-Image retrieval mode
use_lora: true                       # Enable LoRA efficient fine-tuning to reduce VRAM usage

# --- Data Paths ---
data_path: "data/stanford_online_products/train.jsonl"
image_root: "data/stanford_online_products"

# --- Training Parameters ---
epochs: 20
batch_size: 128                      # Larger batch size helps contrastive learning
loss_type: "infonce"                 # Use InfoNCE Loss
learning_rate: 5.0e-5
```

### 4.2 Start Training

After confirming the configuration, execute the following command to start training:

```bash
bash examples/run_dinov2_i2i.sh
```

Or use the CLI command:

```bash
uv run vembed train --config examples/dinov2_i2i.yaml
```

During training, model weights and logs will be saved in the `output_sop_dinov2_i2i` directory.

## 5. Performance Evaluation

After training is complete, we need to evaluate the model's retrieval performance on the SOP test set.

Use the built-in evaluation script:

```bash
# --model_path points to the fine-tuned model checkpoint path
python benchmark/run.py sop \
    --model_path output_sop_dinov2_i2i/checkpoint-1000 \
    --sop_root data/stanford_online_products \
    --batch_size 128
```

### Expected Results

Based on our experimental results on **DINOv2-base**, the performance improvement brought by fine-tuning is significant:

| Model | Metric | Zero-shot (Un-tuned) | Fine-tuned | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv2-base** | **Recall@1** | 55.01% | **84.49%** | **+29.48%** |
| | Recall@10 | 71.09% | 94.00% | +22.91% |

Even using the smaller parameter **DINOv2-small**, Recall@1 can typically reach over **75%**. This indicates that the fine-tuned model can more accurately identify fine-grained product features.

## 6. Summary

Through **vembed-factory**, we have greatly simplified the fine-tuning process for visual Embedding models:
1. Standardize data format using `prepare_sop_data.py`.
2. Flexibly configure model parameters via `dinov2_i2i.yaml`.
3. One-click start for training and evaluation.

This makes building industrial-grade high-precision image search systems more efficient and convenient.

If you are interested in this project, welcome to visit the GitHub repository and give it a Star: [vembed-factory](https://github.com/fangzhensheng/vembed-factory)
