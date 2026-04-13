"""Test VEmbedTrainer configuration and execution."""

from unittest.mock import patch

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
    """Test that loss_type is correctly passed to train_entrypoint config."""
    trainer = VEmbedTrainer(
        model_name="openai/clip-vit-base-patch32",
        loss_type="my_custom_loss",
    )

    with (
        patch("vembed.trainer.load_base_config", return_value={}),
        patch("vembed.trainer.prepare_output_dir") as mock_prepare_output_dir,
        patch("vembed.trainer.train_entrypoint") as mock_train_entrypoint,
    ):
        mock_train_entrypoint.return_value = {
            "output_dir": trainer.output_dir,
            "model_path": "dummy/model/path",
        }

        result = trainer.train(data_path="dummy/path")

        assert result["output_dir"] == trainer.output_dir
        mock_prepare_output_dir.assert_called_once()
        passed_config = mock_train_entrypoint.call_args[0][0]
        assert passed_config["loss_type"] == "my_custom_loss"


def test_custom_collator_config_override():
    """Test that collator_type is correctly passed via encoder_mode override."""
    trainer = VEmbedTrainer(
        model_name="openai/clip-vit-base-patch32",
        collator_type="my_custom_collator",
    )

    with (
        patch("vembed.trainer.load_base_config", return_value={}),
        patch("vembed.trainer.prepare_output_dir"),
        patch("vembed.trainer.train_entrypoint") as mock_train_entrypoint,
    ):
        mock_train_entrypoint.return_value = {
            "output_dir": trainer.output_dir,
            "model_path": "dummy/model/path",
        }

        trainer.train(data_path="dummy/path")

        passed_config = mock_train_entrypoint.call_args[0][0]
        assert passed_config["encoder_mode"] == "my_custom_collator"


def test_default_config_behavior():
    """Test that default values are used when no custom types provided."""
    trainer = VEmbedTrainer(model_name="openai/clip-vit-base-patch32")

    with (
        patch("vembed.trainer.load_base_config", return_value={}),
        patch("vembed.trainer.prepare_output_dir"),
        patch("vembed.trainer.train_entrypoint") as mock_train_entrypoint,
    ):
        mock_train_entrypoint.return_value = {
            "output_dir": trainer.output_dir,
            "model_path": "dummy/model/path",
        }

        trainer.train(data_path="dummy/path")

        passed_config = mock_train_entrypoint.call_args[0][0]
        assert "loss_type" not in passed_config
        assert "encoder_mode" not in passed_config


def test_image_root_passed_to_config():
    """Test that image_root is correctly passed to train_entrypoint config."""
    trainer = VEmbedTrainer(model_name="openai/clip-vit-base-patch32")

    with (
        patch("vembed.trainer.load_base_config", return_value={}),
        patch("vembed.trainer.prepare_output_dir"),
        patch("vembed.trainer.train_entrypoint") as mock_train_entrypoint,
    ):
        mock_train_entrypoint.return_value = {
            "output_dir": trainer.output_dir,
            "model_path": "dummy/model/path",
        }

        trainer.train(data_path="dummy/path", image_root="/data/images")

        passed_config = mock_train_entrypoint.call_args[0][0]
        assert passed_config["image_root"] == "/data/images"


def test_image_root_omitted_when_none():
    """Test that image_root is not set in config when not provided."""
    trainer = VEmbedTrainer(model_name="openai/clip-vit-base-patch32")

    with (
        patch("vembed.trainer.load_base_config", return_value={}),
        patch("vembed.trainer.prepare_output_dir"),
        patch("vembed.trainer.train_entrypoint") as mock_train_entrypoint,
    ):
        mock_train_entrypoint.return_value = {
            "output_dir": trainer.output_dir,
            "model_path": "dummy/model/path",
        }

        trainer.train(data_path="dummy/path")

        passed_config = mock_train_entrypoint.call_args[0][0]
        assert "image_root" not in passed_config
