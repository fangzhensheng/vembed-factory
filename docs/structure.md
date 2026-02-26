# Project Structure & Architecture

This guide is for developers who want to understand or contribute to vembed-factory. For end users, refer to [README.md](../README.md).

## Directory Overview

```
vembed-factory/
├── vembed/                 # Main package
├── examples/               # Runnable training configs & launchers
├── notebooks/              # Jupyter tutorials
├── tests/                  # Unit & integration tests
├── docs/                   # Documentation & guides
├── configs/                # Default YAML presets
├── benchmark/              # Benchmarking tools
├── .github/                # CI/CD workflows & issue templates
├── Dockerfile              # GPU container
└── pyproject.toml          # Package metadata
```

---

## Core Package: `vembed/`

### Layered Architecture

```
vembed/
├── __init__.py             # Public API exports (Trainer, Predictor)
├── cli.py                  # CLI entry point & command parsing
├── config.py               # Runtime configuration management
├── trainer.py              # High-level training orchestration
├── inference.py            # High-level inference API
│
├── model/                  # Model layer (abstraction over different architectures)
│   ├── __init__.py
│   ├── base.py             # BaseEmbeddingModel, pool(), disable_kv_cache()
│   ├── modeling.py         # VisualRetrievalModel (facade combining text + image)
│   ├── encoders_factory.py # SimpleTextEncoder, SimpleImageEncoder
│   ├── registry.py         # ModelRegistry for dynamic model loading
│   ├── backbones/          # Model-specific implementations
│   │   ├── auto.py         # AutoModel loader (for dual-encoder models)
│   │   ├── qwen3.py        # Qwen3-VL-Embedding models
│   │   ├── vlm_generic.py  # Generic VLM loader (Qwen2-VL, Yi-VL)
│   │   └── composed.py     # Composed encoders (custom text + image pairs)
│   └── processors/         # Data processors for different models
│       ├── __init__.py
│       ├── auto.py         # AutoProcessor wrapper
│       ├── qwen3_vl.py     # Qwen3-VL processor
│       └── registry.py     # ProcessorRegistry
│
├── data/                   # Data layer (loading & preprocessing)
│   ├── __init__.py
│   ├── dataset.py          # VisualRetrievalDataset (main dataset class)
│   ├── loading.py          # Data loading utilities (JSONL, CSV, Parquet)
│   ├── registry.py         # CollatorRegistry for batch processing
│   └── collators/          # Batch collation logic
│       ├── __init__.py
│       ├── default.py      # Default collator for dual-encoders
│       ├── qwen3_vl.py     # Qwen3-VL specific collator
│       └── registry.py     # Dynamic collator selection
│
├── losses/                 # Loss functions (training objectives)
│   ├── __init__.py
│   ├── factory.py          # LossFactory for building loss functions
│   ├── registry.py         # LossRegistry for dynamic loss selection
│   └── functions/          # Individual loss implementations
│       ├── __init__.py
│       ├── infonce.py      # InfoNCE (contrastive learning)
│       ├── triplet.py      # Triplet loss
│       ├── cosent.py       # CoSENT loss (symmetric)
│       ├── colbert.py      # ColBERT loss (late interaction)
│       ├── mrl.py          # Matryoshka Representation Learning
│       └── distillation.py # Knowledge distillation
│
├── grad_cache/             # Memory-efficient gradient caching
│   ├── __init__.py
│   ├── grad_cache.py       # Core GradCache implementation (model-agnostic)
│   ├── functional.py       # Functional API for GradCache
│   ├── loss.py             # Loss wrapper for GradCache
│   ├── context_managers.py # RNG state management (RandContext)
│   ├── cachex/             # Extended cache functionality
│   │   ├── functional.py   # CacheX functional API
│   │   ├── training.py     # CacheX training utilities
│   │   └── tree_utils.py   # Nested tensor utilities
│   └── pytorch_lightning/  # PyTorch Lightning integration
│       ├── pl_gradcache.py # Lightning wrapper
│       └── pl_example.py   # Lightning example
│
├── training/               # Training utilities
│   ├── __init__.py
│   ├── gradient_cache_trainer.py  # GradCache trainer implementation
│   └── dpo.py              # Direct Preference Optimization (future)
│
├── entrypoints/            # CLI entry points
│   ├── __init__.py
│   ├── train.py            # Training CLI (main entry point)
│   ├── evaluate.py         # Evaluation with metrics
│   └── evaluate_simple.py  # Simple evaluation utility
│
├── evaluation/             # Evaluation metrics & reporting
│   ├── __init__.py
│   ├── metrics.py          # Recall@K, MRR, NDCG, etc.
│   └── report.py           # HTML/markdown report generation
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── logging.py          # Logging configuration
│   ├── download.py         # Model & dataset downloading
│   └── distributed.py      # Distributed training utilities
│
└── core/                   # (Legacy) Core utilities
    ├── __init__.py
    └── gradient_cache.py   # (Deprecated - use vembed.grad_cache)
```

---

## Key Components Explained

### 1. Model Layer (`vembed/model/`)

**Purpose**: Abstraction layer over different model architectures (CLIP, Qwen-VL, ColPali, etc.)

**Key Classes**:
- `BaseEmbeddingModel` - Base class for all embedding models
- `VisualRetrievalModel` - Combines text encoder + image encoder
- `ModelRegistry` - Factory for loading models by name

**Example Flow**:
```python
# User calls
trainer = Trainer("openai/clip-vit-base-patch32")

# Internally:
# 1. ModelRegistry.get("openai/clip-vit-base-patch32")
# 2. Loads from vembed/model/backbones/auto.py
# 3. Returns BaseEmbeddingModel instance
```

**Adding New Models**:
1. Add model backend to `vembed/model/backbones/`
2. Register in `ModelRegistry`
3. Add processor to `vembed/model/processors/`

### 2. Data Layer (`vembed/data/`)

**Purpose**: Unified data loading from various formats (JSONL, CSV, Parquet, HuggingFace)

**Key Classes**:
- `VisualRetrievalDataset` - Main dataset class
- `Collator` - Batch processing & tokenization
- `CollatorRegistry` - Selects collator based on model type

**Supported Formats**:
```
JSONL: {"query": "...", "positive": "...", "negatives": [...]}
CSV:   With flexible column mapping
HF:    Any HuggingFace dataset
```

**Adding New Data Format**:
1. Implement `load_data()` function in `vembed/data/loading.py`
2. Update `VisualRetrievalDataset` to handle format
3. Register in dataset factory

### 3. Loss Functions (`vembed/losses/`)

**Purpose**: Collection of training objectives for different tasks

**Available Losses**:
- `InfoNCE` - Standard contrastive learning (most common)
- `Triplet` - Triplet loss with margin
- `CoSENT` - Symmetric loss
- `ColBERT` - Late interaction (fine-grained matching)
- `MRL` - Matryoshka learning (multi-scale representation)
- `Distillation` - Knowledge distillation

**Adding New Loss**:
1. Implement in `vembed/losses/functions/`
2. Register in `LossRegistry`
3. Update config schema

### 4. Gradient Cache (`vembed/grad_cache/`)

**Purpose**: Memory-efficient training for large batch sizes

**How It Works**:
```
Forward Pass (Chunk 1) [no_grad]
    ↓
Backward Pass (Chunk 1) [with_grad]
    ↓
Forward Pass (Chunk 2) [no_grad]
    ↓
Backward Pass (Chunk 2) [with_grad]
    ↓
Optimizer Step (gradient accumulated)
```

**Key Features**:
- RNG state restoration for deterministic recomputation
- Compatible with gradient checkpointing
- DDP multi-GPU support via `no_sync()`
- Model-agnostic (works with any PyTorch model)

**See**: [docs/COEXISTENCE_FIX_GUIDE.md](COEXISTENCE_FIX_GUIDE.md) for gradient_cache + gradient_checkpointing interaction.

### 5. Training (`vembed/entrypoints/train.py`)

**Main Training Flow**:
1. Load config (YAML + CLI overrides)
2. Initialize model, tokenizer, processor
3. Load training data (with fallback for None fields)
4. Create optimizer & scheduler
5. Training loop with validation
6. Save checkpoint & final model

**Key Configuration**:
- `use_gradient_cache` - Enable memory optimization
- `gradient_cache_chunk_size` - How many samples per chunk
- `gradient_checkpointing` - Enable activation recomputation
- `use_lora` - Parameter-efficient fine-tuning
- `use_mrl` - Matryoshka learning for multi-scale outputs

---

## Configuration System (`vembed/config.py`)

**Hierarchy** (lowest to highest priority):
```
1. Hardcoded defaults (vembed/config.py)
2. Preset YAML (configs/clip.yaml, configs/qwen3.yaml, etc.)
3. User-provided YAML (examples/my_config.yaml)
4. CLI overrides (--batch_size 64)
```

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | N/A | HuggingFace model ID |
| `batch_size` | int | 32 | Training batch size per GPU |
| `learning_rate` | float | 2e-5 | AdamW learning rate |
| `epochs` | int | 3 | Number of training epochs |
| `use_gradient_cache` | bool | true | Enable memory optimization |
| `gradient_cache_chunk_size` | int | 32 | Chunk size for GradCache |
| `use_lora` | bool | false | Enable LoRA |
| `use_mrl` | bool | false | Enable Matryoshka learning |

See [configs/defaults.yaml](../configs/defaults.yaml) for complete list.

---

## Testing Structure (`tests/`)

```
tests/
├── unit/              # Unit tests (single component)
├── integration/       # Integration tests (multiple components)
└── inference/         # Inference tests
```

**Running Tests**:
```bash
make test              # Run all tests
make uv-test-cov       # With coverage report
```

---

## Example Configs (`configs/` & `examples/`)

**Preset Configs**:
- `configs/defaults.yaml` - Base config for all models
- `configs/clip.yaml` - CLIP-specific overrides
- `configs/qwen3.yaml` - Qwen3-VL-specific overrides
- `configs/siglip.yaml` - SigLIP-specific overrides

**Example Launchers**:
- `examples/clip_train.yaml` - Ready-to-run CLIP training
- `examples/qwen3_2b_train.yaml` - Qwen3-VL-2B training (optimized)
- `examples/qwen_colbert.yaml` - ColPali/ColQwen setup

---

## Contributing Guidelines

### Code Style
- Python 3.10+
- Black for formatting
- Ruff for linting
- Type hints required for new public APIs

### Adding a New Feature

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Implement the feature** with:
   - Type hints
   - Docstrings (especially for public APIs)
   - Unit tests in `tests/unit/`
   - Integration tests if multi-component

3. **Update documentation**:
   - Add docstring to functions/classes
   - Update relevant guide in `docs/`
   - Add example if needed

4. **Run quality checks**:
   ```bash
   make uv-format   # Auto-format
   make uv-lint     # Check style
   make uv-test     # Run tests
   ```

5. **Submit PR** with:
   - Clear description of changes
   - Reference related issues
   - Before/after comparison if applicable

### Common Tasks

**Adding a New Model**:
1. Implement backend in `vembed/model/backbones/`
2. Add processor in `vembed/model/processors/`
3. Register in `ModelRegistry` & `ProcessorRegistry`
4. Add example config in `configs/`
5. Add unit test in `tests/unit/`

**Adding a New Loss Function**:
1. Implement in `vembed/losses/functions/`
2. Register in `LossRegistry`
3. Add tests
4. Document in example config

**Fixing a Bug**:
1. Create test that reproduces bug
2. Fix implementation
3. Verify test passes
4. Add regression test if needed

---

## Development Workflow

### Setup
```bash
git clone https://github.com/fangzhensheng/vembed-factory.git
cd vembed-factory
uv sync                    # Install deps + dev tools
source .venv/bin/activate
```

### Daily Workflow
```bash
# Make changes to code
vim vembed/model/base.py

# Run tests locally
make uv-test

# Format & lint
make uv-format
make uv-lint

# Commit
git add vembed/model/base.py
git commit -m "Fix: Handle None values in batch processing"

# Push & create PR
git push origin feature/amazing-feature
```

### Debugging
```bash
# Run specific test with output
python -m pytest tests/unit/test_model.py::test_load_model -v -s

# Check code style
make uv-lint --verbose

# Check test coverage
make uv-test-cov
```

---

## Performance Considerations

### Memory Optimization

The project implements three-layer memory optimization:

1. **Gradient Cache** (up to 40% VRAM savings)
   - Chunks large batches into smaller forward passes
   - Trades compute for memory

2. **Gradient Checkpointing** (up to 20% VRAM savings)
   - Recomputes activations during backward pass
   - Trades compute for memory

3. **LoRA** (up to 10% VRAM savings)
   - Only trains low-rank adapter layers
   - Significantly reduces parameter count

### Training Speed

For Qwen3-VL-2B with batch_size=128:
- **No optimization**: ~100ms/step, 45GB VRAM (OOM)
- **Gradient Cache**: ~115ms/step, 25-30GB VRAM ✓
- **Full Optimization**: ~120ms/step, 20GB VRAM ✓

---

## References

- [GradCache Paper](https://arxiv.org/abs/2101.06983) - Memory-efficient contrastive learning
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - Model backbone
- [Accelerate](https://huggingface.co/docs/accelerate/) - Distributed training
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) - LoRA implementation

---

## Getting Help

- **Questions**: Open an [Issue](https://github.com/fangzhensheng/vembed-factory/issues) with `[Question]` prefix
- **Bug Reports**: Use [bug report template](https://github.com/fangzhensheng/vembed-factory/issues/new?template=bug_report.md)
- **Feature Requests**: Open an [Issue](https://github.com/fangzhensheng/vembed-factory/issues/new) with `[Feature]` prefix
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)
