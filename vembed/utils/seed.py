"""Random seed management for reproducibility."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(
    seed: int | None = None, workers: bool = False, deterministic: bool = False
) -> None:
    """Set random seeds for reproducibility across different libraries.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - HuggingFace Transformers (if available)

    Args:
        seed: Random seed value. If None, uses a random seed.
        workers: If True, also configure DataLoader workers for determinism.
            This is useful for distributed training.
        deterministic: If True, force deterministic CUDA operations (may reduce
            performance by 5-15%). Use only when absolute reproducibility is required.

    Example:
        >>> set_seed(42)  # Reproducible results with reasonable performance
        >>> set_seed(42, workers=True)  # For multi-GPU training
        >>> set_seed(42, deterministic=True)  # Maximum reproducibility (slower)
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Python's random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch - CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: Force deterministic CUDA operations
    # Note: This can reduce performance by 5-15%, use only when necessary
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic CUDA mode enabled (may reduce performance)")
    else:
        # Allow benchmark optimization for better performance
        torch.backends.cudnn.benchmark = True

    # DataLoader worker seeding for distributed training
    if workers:
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Try to set seed for HuggingFace Transformers if available
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError as e:
        logger.warning("Failed to set HuggingFace seed: %s", e)

    logger.info("Random seed set to %d", seed)
