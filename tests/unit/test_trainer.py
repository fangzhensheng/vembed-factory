"""Test VEmbedTrainer configuration and execution."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from vembed.trainer import VEmbedTrainer


class MockLoss(nn.Module):
    """Mock custom loss for testing."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def forward(self, x, y):
        return (x - y).pow(2).mean()


def test_custom_loss_config_override():
    """Test that loss_type is correctly passed to CLI args."""
    trainer = VEmbedTrainer(
        model_name="openai/clip-vit-base-patch32",
        loss_type="my_custom_loss",
    )

    with patch("vembed.trainer.cli_main") as mock_cli_main:
        mock_cli_main.return_value = 0
        
        # Mock sys.argv to avoid messing up actual sys.argv
        with patch.object(sys, "argv", ["script.py"]):
            trainer.train(data_path="dummy/path")
            
        # Check if cli_main was called
        mock_cli_main.assert_called_once()
        
        # Verify args contain our custom loss type override
        call_args = mock_cli_main.call_args[0][0]
        assert "--config_override" in call_args
        
        # Find index of config override
        override_idx = call_args.index("--config_override")
        overrides = call_args[override_idx+1:]
        
        # Check if loss_type override exists
        assert "loss_type=my_custom_loss" in overrides


def test_custom_collator_config_override():
    """Test that collator_type is correctly passed to CLI args via encoder_mode override."""
    trainer = VEmbedTrainer(
        model_name="openai/clip-vit-base-patch32",
        collator_type="my_custom_collator",
    )

    with patch("vembed.trainer.cli_main") as mock_cli_main:
        mock_cli_main.return_value = 0
        
        with patch.object(sys, "argv", ["script.py"]):
            trainer.train(data_path="dummy/path")
            
        mock_cli_main.assert_called_once()
        
        call_args = mock_cli_main.call_args[0][0]
        assert "--config_override" in call_args
        
        override_idx = call_args.index("--config_override")
        overrides = call_args[override_idx+1:]
        
        # Check if encoder_mode override exists (collator uses encoder_mode key)
        assert "encoder_mode=my_custom_collator" in overrides


def test_default_config_behavior():
    """Test that default values are used when no custom types provided."""
    trainer = VEmbedTrainer(model_name="openai/clip-vit-base-patch32")

    with patch("vembed.trainer.cli_main") as mock_cli_main:
        mock_cli_main.return_value = 0
        
        with patch.object(sys, "argv", ["script.py"]):
            trainer.train(data_path="dummy/path")
            
        call_args = mock_cli_main.call_args[0][0]
        
        # Should contain default overrides but not custom loss/collator
        override_idx = call_args.index("--config_override")
        overrides = call_args[override_idx+1:]
        
        # loss_type should NOT be overridden (defaults to infonce in config loader)
        # unless it was explicitly set to something other than infonce in init
        # In our simplified trainer, default loss_type is "infonce"
        # and logic is: if self.loss_type and self.loss_type != "infonce"
        assert not any(arg.startswith("loss_type=") for arg in overrides)
        
        # encoder_mode should be "auto" (default)
        assert "encoder_mode=auto" in overrides
