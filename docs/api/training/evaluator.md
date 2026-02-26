# Evaluator Module

Utilities for validation and evaluation during training.

**Location**: `vembed/training/evaluator.py`

**Lines**: 130

## Overview

The evaluator module handles:
- Running validation loops
- Computing retrieval metrics (Recall@K, MRR, etc.)
- Logging evaluation results
- Tracking best performance

## Key Classes

### `Evaluator`

```python
class Evaluator:
    """Handles validation loop and metric computation."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        accelerator,
        config: dict | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            criterion: Loss function
            accelerator: Accelerate accelerator
            config: Optional configuration dict
        """

    def evaluate(
        self,
        val_dataloader,
        global_step: int = 0,
    ) -> dict:
        """
        Run validation loop.

        Computes:
        - Average validation loss
        - Retrieval metrics (Recall@K, MRR, NDCG)
        - Other custom metrics if defined

        Args:
            val_dataloader: Validation DataLoader
            global_step: Current training step (for logging)

        Returns:
            dict: Evaluation metrics

        Example:
            {
                "val_loss": 0.25,
                "recall@1": 0.85,
                "recall@10": 0.95,
                "mrr": 0.78,
            }
        """
```

## Usage Example

```python
from vembed.training.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(
    model=model,
    criterion=criterion,
    accelerator=accelerator,
    config=config,
)

# Run validation
metrics = evaluator.evaluate(val_dataloader, global_step=100)

print(f"Validation Loss: {metrics['val_loss']:.4f}")
print(f"Recall@1: {metrics.get('recall@1', 'N/A')}")
print(f"Recall@10: {metrics.get('recall@10', 'N/A')}")
```

## Integration with Training Loop

```python
from vembed.training import Trainer
from vembed.training.evaluator import Evaluator

# During training
evaluator = Evaluator(model, criterion, accelerator)

for epoch in range(num_epochs):
    # Training...
    for step, batch in enumerate(train_dataloader):
        # Train step...
        pass

    # Validation at epoch end
    if eval_dataloader:
        metrics = evaluator.evaluate(
            eval_dataloader,
            global_step=epoch * len(train_dataloader)
        )

        if metrics['val_loss'] < best_loss:
            best_loss = metrics['val_loss']
            # Save checkpoint
            save_checkpoint(...)
```

## Metrics Computed

### Standard Metrics

| Metric | Description |
|--------|-------------|
| `val_loss` | Average validation loss |
| `recall@1` | Fraction of queries with correct item in top-1 |
| `recall@5` | Fraction of queries with correct item in top-5 |
| `recall@10` | Fraction of queries with correct item in top-10 |
| `recall@50` | Fraction of queries with correct item in top-50 |
| `mrr` | Mean Reciprocal Rank (avg rank of first correct item) |
| `ndcg@10` | Normalized Discounted Cumulative Gain |

### Metric Computation

Metrics are computed based on:
- Model embeddings (query vs. candidates)
- Similarity scores (dot product, cosine, etc.)
- Ground truth labels from batch

## Performance Considerations

### Memory Usage
- Stores all query and candidate embeddings
- Scales with validation set size
- Can be problematic for very large validation sets

### Speed
- Typically 10-50x slower than training due to metric computation
- Runs on GPU for efficiency
- Can be disabled if not needed

## Related Modules

- [training_loop.md](./training_loop.md) - Called during Trainer.train()
- [data_utils.md](./data_utils.md) - Batch processing
- [config.md](./config.md) - Evaluation configuration
