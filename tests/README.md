# Test Suite Guide

Comprehensive test suite for vembed-factory with unit and integration tests.

## Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_import.py        # Package import tests
│   ├── test_config.py        # Configuration system tests
│   ├── test_model_builder.py # Model building tests
│   ├── test_optimizer.py     # Optimizer and scheduler tests
│   ├── test_data_pipeline.py # Dataset and dataloader tests
│   ├── test_losses.py        # Loss function tests
│   ├── test_grad_cache.py    # Gradient cache tests
│   ├── test_metrics.py       # Evaluation metrics tests
│   └── ...
├── integration/
│   ├── test_e2e_training.py  # End-to-end training tests
│   └── ...
└── conftest.py               # Pytest configuration
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Run Specific Test Class

```bash
pytest tests/unit/test_config.py::TestDistributedConfig -v
```

### Run Specific Test

```bash
pytest tests/unit/test_config.py::TestDistributedConfig::test_gradient_cache_enabled -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=vembed --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

### Run Tests Matching Pattern

```bash
# Run all tests with "config" in name
pytest tests/ -k "config" -v
```

### Run Tests with Markers

```bash
# Run tests marked as slow
pytest tests/ -m slow -v

# Run tests except slow ones
pytest tests/ -m "not slow" -v
```

## Test Categories

### Unit Tests

**Configuration** (`test_config.py`)
- Config loading
- Config merging
- Distributed config
- Config validation
- YAML config handling

**Model Building** (`test_model_builder.py`)
- Model building (CLIP, etc.)
- Processor loading
- Gradient checkpointing
- Dtype unification for FSDP
- Parameter summary

**Optimizer** (`test_optimizer.py`)
- Optimizer building
- Learning rate configuration
- Scheduler building (cosine, linear, constant)
- Warmup ratio calculation
- Learning rate scheduling

**Data Pipeline** (`test_data_pipeline.py`)
- Dataset creation
- Collator registry
- DataLoader creation
- Batch processing
- Dataset modes (train/eval)

**Loss Functions** (`test_losses.py`)
- InfoNCE loss
- Triplet loss
- CoSENT loss
- Matryoshka (MRL) loss

**Gradient Cache** (`test_grad_cache.py`)
- Gradient cache computation
- Chunk processing
- Backward pass

**Metrics** (`test_metrics.py`)
- Recall@K computation
- MRR (Mean Reciprocal Rank)
- Evaluation metrics

### Integration Tests

**End-to-End Training** (`test_e2e_training.py`)
- Complete training pipeline
- Model building + optimizer + loss
- Dataset + dataloader
- Config loading
- Checkpoint saving

## Key Test Fixtures

### Temporary Data Directory

```python
@pytest.fixture
def temp_data_dir():
    """Creates temp JSONL data with train/val splits."""
```

Usage:
```python
def test_training(temp_data_dir):
    dataset = VisualRetrievalDataset(
        data_source=str(temp_data_dir / "train.jsonl"),
        ...
    )
```

### Temporary Config Directory

```python
@pytest.fixture
def temp_config_dir(temp_data_dir):
    """Creates temp YAML config files."""
```

### Random Seed

```python
@pytest.fixture
def random_seed():
    """Sets reproducible random seed."""
```

## Test Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Config | 90% | 85% |
| Model Builder | 80% | 75% |
| Optimizer | 85% | 80% |
| Data Pipeline | 80% | 70% |
| Loss Functions | 95% | 90% |
| Training Loop | 70% | 60% |

## Adding New Tests

### Template: Unit Test

```python
"""Tests for new_module."""

import pytest
from vembed.new_module import some_function


class TestNewFeature:
    """Test new feature."""

    def test_basic_functionality(self):
        """Test that feature works."""
        result = some_function(input_data)

        assert result is not None
        assert result == expected_output

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            some_function(invalid_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Template: Integration Test

```python
"""Integration tests for training."""

import pytest


@pytest.fixture
def training_setup():
    """Setup training components."""
    # ... setup code ...
    yield config, model, dataloader
    # ... cleanup code ...


class TestTrainingIntegration:
    """Test complete training workflow."""

    def test_full_training_loop(self, training_setup):
        """Test complete training."""
        config, model, dataloader = training_setup

        for batch in dataloader:
            # ... training step ...
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Common Test Patterns

### Testing Configuration

```python
def test_config_merge():
    """Test config merging."""
    base = {"a": 1, "b": 2}
    user = {"b": 3}

    result = merge_configs(base, user)

    assert result["a"] == 1
    assert result["b"] == 3
```

### Testing Model Building

```python
def test_model_building():
    """Test model can be built."""
    config = {"model_name": "openai/clip-vit-base-patch32"}
    model = build_model(config)

    assert model is not None
    assert hasattr(model, "forward")
```

### Testing with Fixtures

```python
def test_dataset_creation(temp_data_dir):
    """Test dataset creation with temp data."""
    dataset = VisualRetrievalDataset(
        data_source=str(temp_data_dir / "train.jsonl"),
        ...
    )

    assert len(dataset) > 0
```

### Testing Loss Computation

```python
def test_loss_backward():
    """Test loss backward pass."""
    embeddings = torch.randn(4, 256)
    loss_fn = InfoNCELoss({"temperature": 0.1})

    loss = loss_fn(embeddings, embeddings, None)
    loss.backward()

    assert torch.isfinite(loss)
```

## Debugging Tests

### Enable Verbose Output

```bash
pytest tests/ -vv  # Very verbose
```

### Show Print Statements

```bash
pytest tests/ -s
```

### Drop into Debugger on Failure

```bash
pytest tests/ --pdb
```

### Show Local Variables on Failure

```bash
pytest tests/ -l
```

## CI/CD Integration

Tests are run automatically on:
- Every push
- Pull requests
- Before releases

Check `.github/workflows/` for CI configuration.

## Performance Testing

### Time Test Execution

```bash
pytest tests/ --durations=10
```

Shows 10 slowest tests.

### Parallel Test Execution

```bash
pip install pytest-xdist
pytest tests/ -n auto  # Uses all CPU cores
```

## Troubleshooting

### Test Failures

1. Check test output: `pytest tests/ -v`
2. Run specific test: `pytest path/to/test.py::TestClass::test_method -v`
3. Use debugger: `pytest --pdb`

### Import Errors

```bash
# Make sure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### GPU Tests

```bash
# Run only CPU tests
pytest tests/ -m "not gpu"

# Run GPU tests
CUDA_VISIBLE_DEVICES=0 pytest tests/ -m gpu
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Test Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Fixtures Guide](https://docs.pytest.org/en/stable/fixture.html)
