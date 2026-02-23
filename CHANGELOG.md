# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Learning rate scheduler support (`cosine`, `linear`, `constant_with_warmup`) with configurable warmup
- Weights & Biases and TensorBoard integration via `--report_to` flag
- Gradient clipping (`max_grad_norm`) for training stability
- Weight decay with proper parameter grouping (no decay on bias/LayerNorm)
- `vembed` CLI entry point (`pip install` then `vembed --help`)
- Pre-commit hooks configuration for consistent code style
- Test coverage reporting with Codecov integration
- Multi-Python version CI matrix (3.10, 3.11, 3.12)
- Optional dependency groups: `demo`, `lora`, `wandb`, `all`
- Package metadata: classifiers, keywords, project URLs

### Changed
- Unified class naming: `VLM2VecTrainer` → `VEmbedFactoryTrainer` (old name kept as alias)
- Upgraded minimum Python version from 3.8 to 3.10
- Improved error handling: replaced bare `except` clauses with specific exception types
- Replaced `print()` logging with `logging` module in core modules
- Docker image now properly installs the `vembed-factory` package

### Fixed
- `vembed/__init__.py` was empty — now exports public API and `__version__`
- `test_import.py` was testing non-existent `src` package
- `quick_start.py` was using deprecated `VLM2VecTrainer` class name

## [0.1.0] - 2025-06-01

### Added
- Initial release
- Support for CLIP, SigLIP, Qwen2-VL fine-tuning
- Gradient Cache for memory-efficient training
- Matryoshka Representation Learning (MRL)
- Late Interaction (ColPali/ColBERT) training
- Composed encoder support (BERT + DINOv2)
- Knowledge distillation support
- LoRA fine-tuning integration
- Gradio demo app
- Jupyter notebook tutorials

[Unreleased]: https://github.com/fangzhensheng/vembed-factory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fangzhensheng/vembed-factory/releases/tag/v0.1.0
