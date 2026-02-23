# vembed-factory

<p align="center">
  <strong>A Factory for Visual & Multimodal Embeddings</strong>
</p>

<p align="center">
  <a href="README_zh-CN.md">中文文档</a> | <a href="README.md">English</a>
</p>

<p align="center">
  <em>Fine-tune CLIP, SigLIP, Qwen-VL and more — for Visual RAG & Multimodal Search.</em>
</p>

<p align="center">
  <a href="https://github.com/fangzhensheng/vembed-factory/actions/workflows/ci.yml"><img src="https://github.com/fangzhensheng/vembed-factory/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://colab.research.google.com/github/fangzhensheng/vembed-factory/blob/main/notebooks/vembed-factory_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

---

## Why vembed-factory?

There are excellent embedding libraries out there. Here's where **vembed-factory** stands:

| Feature | vembed-factory | sentence-transformers | FlagEmbedding |
| :--- | :---: | :---: | :---: |
| **CLIP / SigLIP fine-tuning** | ✅ | Partial | ❌ |
| **Qwen-VL / VLM Embedding** | ✅ | ❌ | ❌ |
| **Gradient Cache (large BS)** | ✅ | ❌ | ✅ |
| **Matryoshka (MRL)** | ✅ | ✅ | ❌ |
| **Late Interaction (ColPali)** | ✅ | ❌ | ❌ |
| **Composed Encoders** | ✅ | ❌ | ❌ |
| **Knowledge Distillation** | ✅ | ✅ | ✅ |
| **LoRA fine-tuning** | ✅ | ❌ | ❌ |
| **W&B / TensorBoard** | ✅ | ✅ | ❌ |
| **Pure training focus** | ✅ | ❌ | ❌ |

**Core philosophy**: *"Do one thing and do it well."* We focus solely on **Training** and **Evaluation**, and output standard HuggingFace weights that you can deploy anywhere — LangChain, Milvus, Vespa, or any vector database.

## Core Features

- **Pure Factory Mode**: Data in -> Fine-tuned Embedding Model + Evaluation Report out.
- **Extensive Model Support**:
  - **Dual-Encoders**: CLIP, SigLIP, EVA-CLIP
  - **Vision-Language Models**: Qwen3-VL-Embedding (2B/8B), Qwen2-VL, and more
  - **Late Interaction**: ColPali, ColQwen (multi-vector fine-grained retrieval)
  - **Composed Encoders**: Mix any Text Encoder + Image Encoder (e.g. BERT + DINOv2)
- **Efficient Training**:
  - **Gradient Cache**: Effective BS=512+ on 16G/24G VRAM GPUs
  - **Matryoshka (MRL)**: One training run -> any embedding dimension (768, 512, 256, 128)
  - **LR Scheduling**: Cosine, linear, constant with warmup
  - **LoRA**: Parameter-efficient fine-tuning
- **Universal Data Engine**: JSONL, CSV, Parquet, HuggingFace Datasets with flexible column mapping
- **Experiment Tracking**: Built-in W&B and TensorBoard integration

## Supported Models

| Model Type | Examples | Use Case |
| :--- | :--- | :--- |
| **Vision-Language Models** | **Qwen3-VL-Embedding-2B/8B**, Qwen2-VL, Yi-VL | SOTA Multimodal Retrieval |
| **Dual-Encoders** | OpenAI CLIP, SigLIP, EVA-CLIP | General Purpose, Zero-Shot |
| **Late Interaction** | **ColPali**, ColQwen | Fine-grained Document Retrieval |
| **Composed** | BERT + DINOv2, BGE + BGE | Specialized Domain, Text Retrieval |


## Quick Start

### Installation

**Option 1: uv (Recommended)**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync (auto-creates .venv, installs all deps from lockfile)
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
uv sync                    # Core + dev tools
source .venv/bin/activate  # Activate the virtual environment
uv sync --all-extras       # (Optional) Include all optional deps (LoRA, W&B, etc.)

# Run commands
uv run python examples/quick_start.py
uv run vembed train --config examples/train_config.yaml
```

**Option 2: pip**

```bash
pip install vembed-factory

# Or install from source
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
pip install -e ".[all]"
```

**Option 3: Docker**

```bash
# Using docker compose (recommended)
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
docker compose up -d
docker compose exec vembed bash

# Or build and run manually
docker build -t vembed-factory .
docker run --gpus all -it -v $(pwd)/data:/app/data vembed-factory bash
```

### Train in 3 Lines (Python API)

```python
from vembed import Trainer

trainer = Trainer("openai/clip-vit-base-patch32")
trainer.train(data_path="data/train.jsonl", output_dir="output", epochs=3)
```

### Inference

```python
from vembed import Predictor

predictor = Predictor(model_path="output/checkpoint-epoch-3")

text_emb = predictor.encode_text("a photo of a cat")
image_emb = predictor.encode_image("cat.jpg")
```

### CLI

```bash
# Train Qwen3-VL-Embedding (Recommended)
vembed --model_type qwen3 --data_path data/train.jsonl

# Train CLIP with MRL
vembed --model_type clip --data_path data/train.jsonl --use_mrl

# Train with W&B logging
vembed --model_type clip --data_path data/train.jsonl \
  --config_override report_to=wandb

# Train ColPali (Late Interaction)
vembed --model_type qwen3 --loss_type colbert --data_path data/train.jsonl
```

## Data Format

vembed-factory supports flexible input data. Each line in a JSONL file is one training example:

| Mode | Example |
| :--- | :--- |
| **Text-to-Image (T2I)** | `{"query": "a cat", "positive": "images/cat.jpg", "negatives": ["images/dog.jpg"]}` |
| **Image-to-Image (I2I)** | `{"query_image": "shoe_query.jpg", "positive": "shoe_target.jpg"}` |
| **Image-to-Text (I2T)** | `{"query_image": "cat.jpg", "positive": "a cute cat on a sofa"}` |
| **Multimodal (M2I/CIR)** | `{"query_image": "shirt_blue.jpg", "query": "change to red", "positive": "shirt_red.jpg"}` |
| **Text-to-Text (T2T)** | `{"query": "capital of france", "positive": "Paris is the capital of France"}` |

## Configuration

Training can be customized via YAML configs, CLI flags, or Python API arguments:

```bash
# Use a preset
vembed --model_type clip --data_path data/train.jsonl

# Override specific settings
vembed --model_type clip --data_path data/train.jsonl \
  --config_override lr=1e-5 batch_size=64 scheduler_type=linear warmup_ratio=0.05
```

Key configuration options:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `scheduler_type` | `cosine` | LR scheduler: `cosine`, `linear`, `constant`, `constant_with_warmup` |
| `warmup_ratio` | `0.1` | Fraction of training steps for warmup |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping (0 to disable) |
| `report_to` | `none` | Experiment tracker: `wandb`, `tensorboard`, `all`, `none` |
| `logging_steps` | `10` | Log metrics every N steps |
| `use_gradient_cache` | `true` | Memory-efficient large batch training |
| `use_mrl` | `false` | Matryoshka Representation Learning |
| `use_lora` | `false` | LoRA fine-tuning |

See [`https://github.com/fangzhensheng/vembed-factory/blob/main/vembed/configs/defaults.yaml`](https://github.com/fangzhensheng/vembed-factory/blob/main/vembed/configs/defaults.yaml) for the full list.

## Project Structure

```
vembed-factory/
├── vembed/
│   ├── __init__.py        # Public API: Trainer, Predictor, __version__
│   ├── trainer.py         # High-level Training API
│   ├── inference.py       # High-level Inference API
│   ├── cli.py             # CLI entry point
│   ├── configs/           # YAML presets (clip, siglip, qwen, ...)
│   ├── model/             # Model backends (CLIP, composed, VLM, ...)
│   ├── losses/            # Loss functions (InfoNCE, MRL, ColBERT, ...)
│   ├── training/          # Gradient Cache implementation
│   └── evaluation/        # Metrics (Recall@K, MRR)
├── examples/              # Scripts, shell launchers, Gradio demo
│   └── benchmark/         # Benchmarking tools
├── notebooks/             # Jupyter tutorials (4 notebooks)
├── tests/                 # Unit tests
├── Dockerfile             # GPU-ready container
└── Makefile               # Common commands (make help)
```

## Development

**Using uv (recommended):**

```bash
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
uv sync                    # Install deps + dev tools
source .venv/bin/activate  # Activate the virtual environment

# Development workflow
make uv-format             # Auto-format code
make uv-lint               # Run linters
make uv-test               # Run tests
make uv-test-cov           # Tests with coverage report
make help                  # Show all available commands
```

**Using pip:**

```bash
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
make install-dev           # Install + pre-commit hooks

make format                # Auto-format code
make lint                  # Run linters
make test                  # Run tests
make docker                # Build Docker image
```

## Benchmark Results

### Stanford Online Products (SOP) - Image-to-Image (I2I)

We fine-tuned **DINOv2-base** and **MAE-base** models on the SOP dataset (e-commerce products) using `vembed-factory`.

**Qualitative Results (Top-5 Retrieval):**
![SOP I2I Demo](assets/sop_i2i_demo.png)

| Model | Metric | Zero-shot | Fine-tuned | Delta (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv2-base** | Recall@1 | 55.01% | **84.49%** | +29.48 |
| *(facebook/dinov2-base)* | Recall@10 | 71.09% | **94.00%** | +22.91 |
| | Recall@100 | 83.95% | **97.74%** | +13.79 |
| **MAE-base** | Recall@1 | 31.28% | **69.08%** | +37.80 |
| *(facebook/vit-mae-base)* | Recall@10 | 46.29% | **84.36%** | +38.06 |
| | Recall@100 | 61.54% | **92.99%** | +31.45 |

**Training Config:**
- **Task**: Image-to-Image Retrieval (I2I)
- **Loss**: Contrastive Learning (InfoNCE)
- **Epochs**: 2 (DINOv2), 3 (MAE)

## Roadmap

- [x] Real benchmark results on Stanford Online Products (SOP)
- [ ] Real benchmark results on Flickr30k / COCO / ViDoRe
- [ ] `sentence-transformers` compatible export format
- [ ] HuggingFace Hub `--push_to_hub` integration
- [ ] ONNX / TorchScript export for production deployment
- [ ] Hard negative mining (cross-batch, offline)
- [ ] Batch inference optimization
- [ ] API documentation site (MkDocs)
- [x] Chinese documentation (中文文档)

## Acknowledgements

- [GradCache](https://github.com/luyug/GradCache) — Memory-efficient contrastive learning
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — Model backends
- [Accelerate](https://github.com/huggingface/accelerate) — Distributed training

## License

MIT License. See [https://github.com/fangzhensheng/vembed-factory/blob/main/LICENSE](https://github.com/fangzhensheng/vembed-factory/blob/main/LICENSE) for details.

## Citation

If you use vembed-factory in your research, please cite:

```bibtex
@misc{vembed-factory,
  author = {Fang Zhensheng},
  title = {vembed-factory: A Factory for Visual & Multimodal Embeddings},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fangzhensheng/vembed-factory}}
}
```
