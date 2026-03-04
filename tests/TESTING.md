# Testing Guide for vembed-factory

## Overview

This project includes comprehensive unit tests for all major components. Tests are organized by module and can be run individually or collectively.

## Quick Start

### Run All Tests

```bash
# Using the convenience script (recommended)
./run_tests.sh --all

# Or using pytest directly
pytest unit -v
```

### Run Specific Test Categories

```bash
# Loss function tests
./run_tests.sh --losses

# Optimizer and scheduler tests
./run_tests.sh --optimizer

# Bidirectional loss tests (new)
./run_tests.sh --bidirectional

# Verbose output for debugging
./run_tests.sh --all -v
```

## Test Structure

```
tests/
├── TESTING.md                       # This file
├── TEST_SUMMARY.md                  # Test results summary
├── run_tests.sh                     # Test runner script
├── conftest.py                      # Shared fixtures and configuration
├── unit/
│   ├── test_bidirectional_loss.py  # ✨ NEW: Bidirectional loss tests
│   ├── test_losses.py              # Loss function tests
│   ├── test_optimizer.py           # Optimizer and scheduler tests
│   ├── test_config.py              # Configuration tests
│   ├── test_dataset.py             # Dataset tests
│   ├── test_metrics.py             # Metrics tests
│   └── ...
├── integration/
│   ├── test_e2e_training.py        # End-to-end training tests
│   └── ...
└── inference/
    └── ...                         # Inference tests
```

## Available Tests

### Unit Tests

#### Loss Functions (`test_losses.py`, `test_bidirectional_loss.py`)

```bash
# Test InfoNCE loss (all variants)
pytest unit/test_losses.py::test_infonce_basic -v
pytest unit/test_bidirectional_loss.py::TestInfoNCEBidirectional -v

# Test Sigmoid loss (all variants)
pytest unit/test_bidirectional_loss.py::TestSigmoidBidirectional -v

# Test bidirectional gradient flow
pytest unit/test_bidirectional_loss.py::TestBidirectionalGradients -v

# Test all loss functions
pytest unit/test_losses.py unit/test_bidirectional_loss.py -v
```

#### Optimizer & Scheduler (`test_optimizer.py`)

```bash
# Test optimizer building
pytest unit/test_optimizer.py::TestOptimizerBuilding -v

# Test scheduler building
pytest unit/test_optimizer.py::TestSchedulerBuilding -v

# Test learning rate scheduling
pytest unit/test_optimizer.py::TestLearningRateScheduling -v
```

#### Other Tests

```bash
# Configuration tests
pytest unit/test_config.py -v

# Dataset tests
pytest unit/test_dataset.py -v

# Metrics tests
pytest unit/test_metrics.py -v
```

## Running Tests Programmatically

```python
import subprocess
import sys

# Run all tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "unit", "-v"],
    cwd="/path/to/vembed-factory/tests"
)
sys.exit(result.returncode)
```

## Test Coverage

### Bidirectional Loss Tests

The new `test_bidirectional_loss.py` includes:

✅ **InfoNCE Tests** (8 test cases)
- Unidirectional mode (default)
- Bidirectional mode
- With 3D negative embeddings
- With supervised contrastive loss
- With in-batch negatives
- Configuration defaults

✅ **Sigmoid Tests** (7 test cases)
- Unidirectional vs bidirectional
- With labels
- With custom logit parameters
- In-batch negatives
- Configuration defaults

✅ **Gradient Tests** (3 test cases)
- InfoNCE gradient flow
- Sigmoid gradient flow
- Logit parameter gradients

✅ **Consistency Tests** (4 test cases)
- Symmetry properties
- No negatives case
- Batch size 1
- Large batch (32 samples)

**Total: 22 bidirectional loss test cases**

## Test Features

### Fixtures (from `conftest.py`)

```python
@pytest.fixture
def device():
    """Automatically use GPU if available, otherwise CPU"""
    pass

@pytest.fixture
def random_seed():
    """Reproducible randomness"""
    pass

@pytest.fixture
def cpu_device():
    """Force CPU device"""
    pass
```

### Test Markers

Tests can be marked with categories:

```python
@pytest.mark.gpu
def test_gpu_only():
    pass

@pytest.mark.slow
def test_slow_training():
    pass
```

Run tests by marker:

```bash
# Run only GPU tests
pytest -m gpu

# Skip slow tests
pytest -m "not slow"
```

## Continuous Integration

All tests should pass before committing:

```bash
# Pre-commit check
./run_tests.sh --all

# If any test fails, investigate and fix:
pytest unit/test_bidirectional_loss.py::TestSigmoidBidirectional::test_sigmoid_bidirectional -vv
```

## Debugging Failed Tests

### Verbose Output

```bash
./run_tests.sh --all -v
# or
pytest unit -vv --tb=long
```

### Run Single Test

```bash
pytest unit/test_bidirectional_loss.py::TestInfoNCEBidirectional::test_infonce_bidirectional -vv
```

### Show Print Statements

```bash
pytest unit/test_bidirectional_loss.py -vv -s
```

### Drop into Debugger

```bash
pytest unit/test_bidirectional_loss.py::TestInfoNCEBidirectional::test_infonce_bidirectional --pdb
```

## Writing New Tests

### Example Test for Bidirectional Loss

```python
def test_my_bidirectional_feature():
    """Test my new bidirectional feature."""
    # Setup
    q = torch.randn(4, 16)
    p = torch.randn(4, 16)
    n = torch.randn(8, 16)

    # Test bidirectional mode
    config = {"loss_bidirectional": True}
    loss_fn = InfoNCELoss(config)
    loss = loss_fn(q, p, n)

    # Assertions
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive"
```

### Common Assertions

```python
# Tensor properties
assert torch.isfinite(loss), "Loss should be finite (no NaN/Inf)"
assert loss.item() > 0, "Loss should be positive"
assert loss.grad is not None, "Loss should have gradients"

# Comparisons
assert torch.allclose(a, b, atol=1e-5), "Tensors should be close"
assert torch.allclose(a.grad, expected_grad, rtol=1e-3), "Gradient check"

# Shapes
assert loss.shape == torch.Size([]), "Loss should be scalar"
assert logits.shape == (batch_size, num_docs), "Logits shape check"
```

## Performance Testing

To benchmark loss computation:

```python
import time

def benchmark_loss():
    q = torch.randn(128, 768, device='cuda')
    p = torch.randn(128, 768, device='cuda')
    n = torch.randn(256, 768, device='cuda')

    config = {"loss_bidirectional": True}
    loss_fn = InfoNCELoss(config)

    start = time.time()
    for _ in range(100):
        loss = loss_fn(q, p, n)
        loss.backward()
    elapsed = time.time() - start

    print(f"100 iterations: {elapsed:.3f}s")
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "module not found"
```bash
# Install in development mode
cd /path/to/vembed-factory
pip install -e .
pytest tests/unit
```

**Issue**: GPU tests fail but CPU tests pass
```bash
# Run CPU-only tests
pytest unit -k "not gpu"
```

**Issue**: Stochastic test failures
```bash
# Use fixed random seed
pytest unit --tb=short
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - run: pip install -e .
      - run: pytest tests/unit -v
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
