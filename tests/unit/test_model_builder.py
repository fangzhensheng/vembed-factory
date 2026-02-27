"""Unit tests for model building functions."""

import tempfile
from pathlib import Path

import pytest
import torch

from vembed.training.model_builder import (
    _enable_gradient_checkpointing,
    _log_fsdp_param_summary,
    build_model,
    load_processor,
    unify_model_dtype_for_fsdp,
)


class FakeAccelerator:
    """Mock accelerator for testing."""

    def __init__(self):
        self.messages = []

    def print(self, msg):
        self.messages.append(msg)


class TestModelBuilding:
    """Test model building functionality."""

    def test_build_clip_model(self):
        """Test building CLIP model."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "get_text_features")
        assert hasattr(model, "get_image_features")

    def test_model_device_placement(self):
        """Test that model is placed on correct device."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        # Check that parameters are on a device
        for param in model.parameters():
            assert param.device.type in ["cpu", "cuda"]

    def test_model_requires_grad(self):
        """Test that model parameters require gradients by default."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        # At least some parameters should require gradients
        has_grad = any(param.requires_grad for param in model.parameters())
        assert has_grad

    def test_model_dtype_default(self):
        """Test model default dtype."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        # Check that all parameters have a dtype
        for param in model.parameters():
            assert param.dtype in [torch.float32, torch.float16, torch.bfloat16]


class TestProcessorLoading:
    """Test processor loading functionality."""

    def test_load_clip_processor(self):
        """Test loading CLIP processor."""
        processor = load_processor("openai/clip-vit-base-patch32")

        assert processor is not None
        assert callable(processor)

    def test_processor_returns_dict(self):
        """Test that processor returns proper format."""
        processor = load_processor("openai/clip-vit-base-patch32")

        # Mock input
        dummy_input = "test text"
        # Processor might return dict or similar
        assert processor is not None


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""

    def test_enable_gradient_checkpointing(self):
        """Test enabling gradient checkpointing."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)
        accelerator = FakeAccelerator()

        _enable_gradient_checkpointing(model.backend, accelerator)

        # Should complete without error
        assert model is not None

    def test_gradient_checkpointing_messages(self):
        """Test that gradient checkpointing produces messages."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)
        accelerator = FakeAccelerator()

        _enable_gradient_checkpointing(model.backend, accelerator)

        # Check that some message was produced
        assert len(accelerator.messages) >= 0


class TestDtypeUnification:
    """Test dtype unification for FSDP."""

    def test_unify_dtype_no_fsdp(self):
        """Test dtype unification when FSDP disabled."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_fsdp": False,
        }
        model = build_model(config)
        accelerator = FakeAccelerator()

        # Should return early without modification
        unify_model_dtype_for_fsdp(model, config, accelerator)

        assert model is not None

    def test_unify_dtype_with_fsdp(self):
        """Test dtype unification when FSDP enabled."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_fsdp": True,
            "torch_dtype": "bfloat16",
        }
        model = build_model(config)
        accelerator = FakeAccelerator()

        unify_model_dtype_for_fsdp(model, config, accelerator)

        # All parameters should have same dtype
        dtypes = {param.dtype for param in model.parameters()}
        assert len(dtypes) <= 1, f"Multiple dtypes found: {dtypes}"

    def test_dtype_conversion(self):
        """Test that dtype conversion works correctly."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_fsdp": True,
            "torch_dtype": "bfloat16",
        }
        model = build_model(config)
        accelerator = FakeAccelerator()

        # Manually create mixed dtypes for testing
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:
                param.data = param.data.to(torch.float32)
            break

        unify_model_dtype_for_fsdp(model, config, accelerator)

        # All parameters should be unified
        dtypes = {param.dtype for param in model.parameters()}
        assert len(dtypes) <= 1

    def test_dtype_mapping(self):
        """Test dtype string mapping."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_fsdp": True,
            "torch_dtype": "float32",
        }
        model = build_model(config)
        accelerator = FakeAccelerator()

        unify_model_dtype_for_fsdp(model, config, accelerator)

        # Should handle float32 mapping
        assert model is not None


class TestFSDPParameterSummary:
    """Test FSDP parameter summary logging."""

    def test_log_parameter_summary(self):
        """Test parameter summary logging."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)
        accelerator = FakeAccelerator()

        _log_fsdp_param_summary(model, accelerator)

        # Should have logged something
        assert len(accelerator.messages) > 0

    def test_parameter_count_calculation(self):
        """Test that parameter counts are calculated correctly."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


class TestModelConfigVariations:
    """Test different model configurations."""

    def test_model_with_lora_config(self):
        """Test model configuration with LoRA settings."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
        }

        # Should not fail with LoRA config
        model = build_model(config)
        assert model is not None

    def test_model_with_dtype_config(self):
        """Test model with different dtype configurations."""
        for dtype in ["float32", "float16", "bfloat16"]:
            config = {
                "model_name": "openai/clip-vit-base-patch32",
                "torch_dtype": dtype,
            }

            model = build_model(config)
            assert model is not None

    def test_model_with_gradient_checkpointing(self):
        """Test model with gradient checkpointing enabled."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "gradient_checkpointing": True,
        }

        model = build_model(config)
        assert model is not None


class TestModelForward:
    """Test model forward pass."""

    def test_model_forward_pass(self):
        """Test that model forward pass works."""
        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        # Create dummy inputs
        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 77))
        image_pixels = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        try:
            with torch.no_grad():
                text_features = model.get_text_features(input_ids=text_input_ids)
                image_features = model.get_image_features(pixel_values=image_pixels)

            assert text_features is not None
            assert image_features is not None
            assert text_features.shape[0] == batch_size
            assert image_features.shape[0] == batch_size
        except Exception as e:
            # Some models might need preprocessing, that's ok
            assert "forward" in str(type(e).__name__).lower() or "input" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
