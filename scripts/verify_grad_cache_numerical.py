import logging

import torch
import torch.nn as nn

from vembed.grad_cache.grad_cache import GradCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            # Note: We disable Dropout for numerical verification against full batch
            # because splitting batches changes the RNG sequence compared to a single full batch pass.
            # The core logic of GradCache (gradient stitching) is best verified deterministically.
            nn.Linear(10, 5),
        )

    def forward(self, x):
        return self.net(x)


def run_verification():
    logger.info("Starting Gradient Cache Numerical Verification...")

    torch.manual_seed(42)
    batch_size = 8
    chunk_size = 2  # 4 chunks
    input_dim = 10

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, 5)

    # Loss function: simple MSE
    loss_fn = nn.MSELoss()

    # Initialize model
    model = ToyModel()

    # Save state to ensure both runs start identical
    state_dict = model.state_dict()

    logger.info("Running Baseline (Full Batch)...")
    model_baseline = ToyModel()
    model_baseline.load_state_dict(state_dict)
    model_baseline.zero_grad()

    # Forward
    pred_baseline = model_baseline(x)
    loss_baseline = loss_fn(pred_baseline, target)

    # Backward
    loss_baseline.backward()

    # Store gradients
    grads_baseline = {k: v.grad.clone() for k, v in model_baseline.named_parameters()}

    logger.info(f"Baseline Loss: {loss_baseline.item():.6f}")

    logger.info("Running Gradient Cache...")
    model_gc = ToyModel()
    model_gc.load_state_dict(state_dict)
    model_gc.zero_grad()

    # GradCache expects a loss function that takes *reps and **kwargs
    def gc_loss_fn(*reps, target_val=None):
        # reps[0] is the output of the model
        return loss_fn(reps[0], target_val)

    gc = GradCache(models=[model_gc], chunk_sizes=chunk_size, loss_fn=gc_loss_fn)

    # Run cache step
    # We pass 'target_val' as a kwarg, which will be passed to gc_loss_fn
    loss_gc = gc.cache_step(x, target_val=target)

    # Store gradients
    grads_gc = {k: v.grad.clone() for k, v in model_gc.named_parameters()}

    logger.info(f"GradCache Loss: {loss_gc.item():.6f}")

    logger.info("\nComparing Gradients...")
    all_passed = True
    max_diff_overall = 0.0

    # Check Loss
    loss_diff = abs(loss_baseline.item() - loss_gc.item())
    if loss_diff > 1e-5:
        logger.error(
            f"Loss mismatch! Baseline: {loss_baseline.item()}, GC: {loss_gc.item()}, Diff: {loss_diff}"
        )
        all_passed = False
    else:
        logger.info(f"Losses match (Diff: {loss_diff:.2e})")

    # Check Gradients
    for k in grads_baseline:
        g_base = grads_baseline[k]
        g_gc = grads_gc[k]

        diff = (g_base - g_gc).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)

        if diff > 1e-5:
            logger.error(f"Gradient mismatch in {k}: Max Diff = {diff:.2e}")
            all_passed = False
        else:
            logger.info(f"Gradient match in {k}: Max Diff = {diff:.2e}")

    if all_passed:
        logger.info("\nVERIFICATION PASSED: Gradient Cache matches Full Batch Backprop.")
    else:
        logger.error("\nVERIFICATION FAILED: Mismatches detected.")


if __name__ == "__main__":
    run_verification()
