"""Training modules for vembed-factory.

This package contains modularized training components:
- config: Configuration management
- data_utils: Data processing utilities
- optimizer_builder: Optimizer and scheduler creation
- model_builder: Model initialization and setup
- checkpoint: Checkpoint management
- evaluator: Evaluation and validation
- training_loop: Core training loop (note: different from vembed.trainer.VEmbedFactoryTrainer)
"""

from vembed.training.config import load_and_parse_config
from vembed.training.training_loop import Trainer

__all__ = [
    "load_and_parse_config",
    "Trainer",
]
