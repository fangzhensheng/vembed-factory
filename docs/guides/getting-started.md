# Getting Started with vembed-factory

vembed-factory is a training framework optimized for visual and multimodal embeddings. This guide shows you how to train your first model in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory

# Install with uv (recommended)
uv sync
source .venv/bin/activate

# Or with pip
pip install -e ".[all]"
```

## Your First Training

### 1. Prepare Data

Create `data/train.jsonl` with retrieval pairs:

```jsonl
{"query": "a red cat", "positive": "cat_red.jpg", "negatives": ["dog.jpg", "cat_blue.jpg"]}
{"query": "a dog running", "positive": "dog_running.jpg", "negatives": ["dog_sitting.jpg"]}
```

Supported formats:
- **Text-to-Image (T2I)**: `{"query": "...", "positive": "image.jpg", "negatives": [...]}`
- **Image-to-Image (I2I)**: `{"query_image": "...", "positive": "...", "negatives": [...]}`
- **Image-to-Text (I2T)**: `{"query_image": "...", "positive": "text"}`
- **Multimodal (M2I)**: `{"query_image": "...", "query": "...", "positive": "..."}`
- **Text-to-Text (T2T)**: `{"query": "...", "positive": "..."}`

### 2. Train with Python API

```python
from vembed import Trainer

# Initialize with any HuggingFace model
trainer = Trainer("openai/clip-vit-base-patch32")

# Train in 3 lines
trainer.train(
    data_path="data/train.jsonl",
    output_dir="output",
    epochs=3
)
```

### 3. Train with CLI

```bash
# Simple training
python run.py examples/clip_train.yaml

# Override parameters
python run.py examples/clip_train.yaml --batch_size 64 --learning_rate 1e-5
```

### 4. Use the Trained Model

```python
from vembed import Predictor

# Load the model
predictor = Predictor("output/checkpoint-epoch-3")

# Encode text
text_emb = predictor.encode_text("a red cat")

# Encode image
image_emb = predictor.encode_image("cat_red.jpg")

# Compute similarity
score = (text_emb * image_emb).sum()
print(f"Similarity: {score:.4f}")
```

## What's Next?

- **[Data Preparation Guide](data-preparation.md)** - Learn about data formats and preprocessing
- **[Configuration Guide](configuration.md)** - Explore training options
- **[Advanced Training](training-advanced.md)** - Gradient caching, LoRA, MRL, etc.

## Supported Models

vembed-factory works with any HuggingFace model, with built-in optimization for:

- **CLIP / SigLIP**: Fast dual-encoders
- **Qwen3-VL-Embedding**: State-of-the-art multimodal models
- **ColPali**: Document and fine-grained retrieval
- **Custom models**: Mix any text + image encoder

## Common Issues

**Q: Out of memory?**
```bash
python run.py config.yaml --batch_size 8 --use_gradient_cache
```

**Q: How to log to W&B?**
```bash
wandb login
python run.py config.yaml --report_to wandb
```

See [Troubleshooting](../troubleshooting.md) for more help.
