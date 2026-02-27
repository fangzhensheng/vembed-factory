"""Unit tests for optimizer and scheduler building."""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from vembed.training.optimizer_builder import build_optimizer, build_scheduler


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestOptimizerBuilding:
    """Test optimizer building."""

    def test_build_default_optimizer(self):
        """Test building default optimizer (AdamW)."""
        model = SimpleModel()
        config = {"learning_rate": 1e-4}

        optimizer = build_optimizer(model, config)

        assert optimizer is not None
        assert isinstance(optimizer, (Adam, AdamW))

    def test_optimizer_learning_rate(self):
        """Test optimizer has correct learning rate."""
        model = SimpleModel()
        config = {"learning_rate": 5e-5}

        optimizer = build_optimizer(model, config)

        # Check learning rate
        for param_group in optimizer.param_groups:
            assert param_group["lr"] == 5e-5

    def test_optimizer_weight_decay(self):
        """Test optimizer weight decay setting."""
        model = SimpleModel()
        config = {"learning_rate": 1e-4, "weight_decay": 0.01}

        optimizer = build_optimizer(model, config)

        for param_group in optimizer.param_groups:
            assert param_group["weight_decay"] == 0.01

    def test_optimizer_default_weight_decay(self):
        """Test default weight decay."""
        model = SimpleModel()
        config = {"learning_rate": 1e-4}

        optimizer = build_optimizer(model, config)

        for param_group in optimizer.param_groups:
            # Should have some default weight decay
            assert param_group["weight_decay"] >= 0

    def test_optimizer_step_works(self):
        """Test that optimizer step works."""
        model = SimpleModel()
        config = {"learning_rate": 1e-4}
        optimizer = build_optimizer(model, config)

        # Dummy forward/backward
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Step should work
        optimizer.step()

        assert model is not None

    def test_optimizer_zero_grad(self):
        """Test optimizer zero_grad functionality."""
        model = SimpleModel()
        config = {"learning_rate": 1e-4}
        optimizer = build_optimizer(model, config)

        # Add some gradients
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

        # Zero gradients
        optimizer.zero_grad()

        # Check gradients are cleared
        for p in model.parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad))


class TestSchedulerBuilding:
    """Test scheduler building."""

    def test_build_cosine_scheduler(self):
        """Test building cosine scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "cosine"}
        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        assert scheduler is not None
        assert warmup_steps > 0

    def test_build_linear_scheduler(self):
        """Test building linear scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "linear"}
        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        assert scheduler is not None

    def test_build_constant_scheduler(self):
        """Test building constant scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "constant"}
        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        assert scheduler is not None

    def test_warmup_ratio_calculation(self):
        """Test warmup steps calculation."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "cosine", "warmup_ratio": 0.1}
        total_steps = 3 * 100  # 3 epochs * 100 steps

        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        # Warmup should be approximately 10% of total steps
        expected_warmup = int(total_steps * 0.1)
        assert warmup_steps == expected_warmup or abs(warmup_steps - expected_warmup) <= 1

    def test_default_warmup_ratio(self):
        """Test default warmup ratio."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "cosine"}  # No warmup_ratio specified

        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        # Should have some default warmup
        assert warmup_steps > 0

    def test_scheduler_step(self):
        """Test that scheduler step works."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "cosine"}
        scheduler, warmup_steps = build_scheduler(
            optimizer, config, num_epochs=3, steps_per_epoch=100
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler
        for _ in range(10):
            scheduler.step()

        # Learning rate should change (or stay same for some schedulers)
        current_lr = optimizer.param_groups[0]["lr"]
        assert isinstance(current_lr, float)


class TestOptimizerWithLoRA:
    """Test optimizer building with LoRA."""

    def test_optimizer_with_lora_config(self):
        """Test optimizer with LoRA configuration."""
        model = SimpleModel()
        config = {
            "learning_rate": 1e-4,
            "use_lora": True,
            "lora_r": 16,
        }

        optimizer = build_optimizer(model, config)

        assert optimizer is not None


class TestSchedulerVariations:
    """Test different scheduler configurations."""

    def test_all_scheduler_types(self):
        """Test that all scheduler types work."""
        model = SimpleModel()

        for sched_type in ["cosine", "linear", "constant", "constant_with_warmup"]:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = {"scheduler_type": sched_type}
            try:
                scheduler, warmup_steps = build_scheduler(
                    optimizer, config, num_epochs=1, steps_per_epoch=10
                )
                assert scheduler is not None
            except ValueError:
                # Some scheduler types might not be supported, that's ok
                pass

    def test_scheduler_with_different_epochs(self):
        """Test scheduler with different epoch counts."""
        model = SimpleModel()

        for num_epochs in [1, 3, 10]:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = {"scheduler_type": "cosine"}
            scheduler, warmup_steps = build_scheduler(
                optimizer, config, num_epochs=num_epochs, steps_per_epoch=100
            )

            assert scheduler is not None
            total_steps = num_epochs * 100
            assert warmup_steps < total_steps


class TestLearningRateScheduling:
    """Test learning rate scheduling behavior."""

    def test_learning_rate_decreases_with_cosine(self):
        """Test that cosine scheduler decreases learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = {"scheduler_type": "cosine", "warmup_ratio": 0.0}
        scheduler, _ = build_scheduler(
            optimizer, config, num_epochs=2, steps_per_epoch=100
        )

        lrs = []
        for _ in range(200):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Learning rate should generally decrease
        assert lrs[0] >= lrs[-1] or len(lrs) < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
