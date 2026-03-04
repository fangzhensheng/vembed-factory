"""Pytest configuration and fixtures."""

import os

import pytest


def pytest_configure(config):
    """Configure pytest."""
    # Set environment variables for testing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session", autouse=True)
def init_accelerate():
    """Initialize accelerate state for logging."""
    from accelerate import Accelerator

    # Initialize with cpu to avoid DDP checks in tests
    Accelerator(cpu=True)


@pytest.fixture(autouse=True)
def mock_huggingface(monkeypatch):
    """Mock HuggingFace model/processor loading."""
    from unittest.mock import MagicMock

    import torch

    # Mock Processor
    processor = MagicMock()
    processor.image_processor = MagicMock()
    # Mock return value for image processing
    processor.image_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    # Mock tokenizer
    processor.tokenizer = MagicMock()
    processor.tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    processor.tokenizer.pad_token_id = 0

    # Mock __call__
    def processor_call(*args, **kwargs):
        if "images" in kwargs:
            return {"pixel_values": torch.randn(1, 3, 224, 224)}
        return {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

    processor.side_effect = processor_call

    # Mock Model
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 768
    model.config.projection_dim = 512

    # Mock parameters for FSDP/dtype checks
    param = MagicMock()
    param.dtype = torch.float32
    param.data = torch.randn(10, 10)
    param.requires_grad = True
    param.numel.return_value = 100
    model.parameters.return_value = [param]

    # Apply patches
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained", lambda *args, **kwargs: processor
    )
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr("transformers.CLIPModel.from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        "transformers.CLIPProcessor.from_pretrained", lambda *args, **kwargs: processor
    )


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
