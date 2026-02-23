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
    <a href="https://github.com/topics/multimodal-embedding">
        <img src="https://img.shields.io/badge/Topic-Multimodal_Embedding-blue.svg" alt="Topic: Multimodal Embedding">
    </a>
    <a href="https://github.com/topics/visual-rag">
        <img src="https://img.shields.io/badge/Topic-Visual_RAG-green.svg" alt="Topic: Visual RAG">
    </a>
    <a href="https://github.com/topics/contrastive-learning">
        <img src="https://img.shields.io/badge/Topic-Contrastive_Learning-orange.svg" alt="Topic: Contrastive Learning">
    </a>
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
# Train using a YAML config (Recommended)
python run.py examples/qwen3_2b_train.yaml

# Train CLIP with MRL
python run.py examples/clip_train.yaml

# Train ColPali (Late Interaction)
python run.py examples/qwen_colbert.yaml
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

Training is configured primarily via YAML files. We follow a hierarchy:
`Defaults` < `Preset YAML` < `User YAML` < `CLI Overrides`.

```bash
# Run with a specific config file
python run.py examples/clip_train.yaml

# Override specific settings via CLI
python run.py examples/clip_train.yaml --batch_size 64 --learning_rate 1e-5
```

Key configuration options (can be set in YAML or CLI):

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

See [`configs/defaults.yaml`](configs/defaults.yaml) for the full list of available parameters.

## Project Structure

```
vembed-factory/
├── configs/               # Base YAML presets (defaults.yaml, clip.yaml, qwen3.yaml, ...)
├── examples/              # Runnable examples with specific YAML configs
├── vembed/
│   ├── __init__.py            # Public API
│   ├── cli.py                 # CLI entry point logic
│   ├── hparams.py             # Configuration dataclasses
│   ├── trainer.py             # High-level Training API
│   ├── inference.py           # High-level Inference API
│   ├── model/                 # Model layer
│   │   ├── backbones/         # Model backends (auto, composed, qwen3, vlm_generic)
│   │   ├── processors/        # Processor registry (AutoProcessor, Qwen3VLProcessor, ...)
│   │   ├── base.py            # BaseEmbeddingModel, pool()
│   │   ├── modeling.py        # VisualRetrievalModel facade
│   │   ├── encoders_factory.py # SimpleTextEncoder, SimpleImageEncoder
│   │   └── registry.py        # ModelRegistry
│   ├── data/                  # Data layer
│   │   ├── collators/         # Collator registry (default, qwen3_vl)
│   │   ├── dataset.py         # VisualRetrievalDataset
│   │   ├── loading.py         # Data loading utilities
│   │   └── registry.py        # CollatorRegistry
│   ├── losses/                # Loss functions
│   │   ├── functions/         # InfoNCE, MRL, ColBERT, triplet, cosent, distillation
│   │   ├── factory.py         # Loss builder
│   │   └── registry.py        # LossRegistry
│   ├── grad_cache/            # Core GradCache library (model-agnostic)
│   ├── training/              # Training utilities (GradCache wrapper with VLM support)
│   ├── entrypoints/           # CLI entrypoints (train, evaluate, evaluate_simple)
│   ├── evaluation/            # Metrics (Recall@K, MRR) & report generation
│   └── utils/                 # Shared utilities
├── benchmark/                 # Standalone benchmarking tools & dataset adapters
├── examples/                  # Shell launchers & YAML configs for all model types
├── notebooks/                 # Jupyter tutorials (5 notebooks)
├── tests/                     # Unit tests
├── docs/                      # MkDocs documentation source
├── Dockerfile                 # GPU-ready container
├── docker-compose.yaml        # Docker Compose setup
├── Makefile                   # Common commands (make help)
└── pyproject.toml             # Project metadata & dependencies
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
![SOP I2I Demo](docs/assets/sop_i2i_demo.png)

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

### Flickr30k - Text↔Image Retrieval (T2I / I2T)

We fine-tuned **CLIP ViT-B/32** on Flickr30k (Karpathy split) using `vembed-factory`.

**Qualitative Results (Top-5 Retrieval):**
![Flickr30k T2I Demo](docs/assets/flickr30k_t2i_demo.png)

#### Text → Image

| Model | Metric | Zero-shot | Fine-tuned | Delta (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP ViT-B/32** | Recall@1 | 58.12% | **70.54%** | +12.42 |
| *(openai/clip-vit-base-patch32)* | Recall@5 | 83.18% | **91.82%** | +8.64 |
| | Recall@10 | 89.46% | **95.80%** | +6.34 |
| | MRR | 69.23% | **79.79%** | +10.56 |
| **Qwen3-VL-2B** | Recall@1 | 83.26% | **85.84%** | +2.58 |
| *(Qwen/Qwen3-VL-Embedding-2B)* | Recall@5 | 96.04% | **97.38%** | +1.34 |
| | Recall@10 | 98.04% | **98.78%** | +0.74 |
| | MRR | 89.02% | **91.02%** | +2.00 |

#### Image → Text

| Model | Metric | Zero-shot | Fine-tuned | Delta (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP ViT-B/32** | Recall@1 | 78.60% | **80.70%** | +2.10 |
| *(openai/clip-vit-base-patch32)* | Recall@5 | 95.40% | **96.00%** | +0.60 |
| | Recall@10 | 98.40% | **98.80%** | +0.40 |
| | MRR | 85.80% | **87.25%** | +1.45 |
| **Qwen3-VL-2B** | Recall@1 | 94.10% | **94.70%** | +0.60 |
| *(Qwen/Qwen3-VL-Embedding-2B)* | Recall@5 | 99.60% | **99.80%** | +0.20 |
| | Recall@10 | 100.00% | **100.00%** | +0.00 |
| | MRR | 96.48% | **96.87%** | +0.39 |

**Rsum Comparison**:
- CLIP ViT-B/32: 503.16 → **533.66** (+30.50)
- Qwen3-VL-2B: 571.04 → **576.50** (+5.46)

### Matryoshka Representation Learning (MRL)

We fine-tuned **Qwen3-VL-Embedding-2B** with MRL enabled, allowing the model to produce high-quality embeddings at variable dimensions (from 1536 down to 256).

**Flickr30k Text-to-Image (T2I) Results:**

| Dimension | Metric | Zero-shot | Fine-tuned (MRL) | Delta (pp) |
| :--- | :--- | :--- | :--- | :--- |
| **1536 (Full)** | Recall@1 | 83.24% | **84.98%** | +1.74 |
| | Recall@5 | 96.12% | **96.66%** | +0.54 |
| **1024** | Recall@1 | 82.74% | **84.68%** | +1.94 |
| **768** | Recall@1 | 82.18% | **84.42%** | +2.24 |
| **512** | Recall@1 | 81.80% | **84.50%** | +2.70 |
| **256** | Recall@1 | 80.18% | **83.20%** | +3.02 |

**Key Insight:**
- **6x Compression**: At 256 dimensions, the MRL-tuned model achieves **83.20%** Recall@1, matching the performance of the full 1536-dimensional original model (83.24%).
- **Adaptive**: One single model supports all these dimensions. You can store 256-dim vectors for fast search and 1536-dim vectors for high-precision reranking.

**Training Config:**
- **Task**: Text↔Image Retrieval (T2I + I2T)
- **Loss**: Contrastive Learning (InfoNCE)
- **Model**: Qwen3-VL-Embedding-2B

## Roadmap

- [x] Real benchmark results on Stanford Online Products (SOP)
- [x] Real benchmark results on Flickr30k
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

MIT License. See [LICENSE](LICENSE) for details.

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
