"""Pytest configuration and fixtures."""

import os

import pytest


def pytest_configure(config):
    """Configure pytest."""
    # Set environment variables for testing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def cpu_device():
    """Provide CPU device for tests."""
    import torch

    return torch.device("cpu")


@pytest.fixture(scope="session")
def device():
    """Provide appropriate device for tests."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    import random

    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return seed
