"""Tests for benchmark model_wrapper improvements.

Tests the new VEmbedWrapper features including:
- encoder_mode auto-detection
- text-only model support (qwen)
- multimodal model support (qwen-vl)
- supports_images property
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402
import torch  # noqa: E402

from benchmark.model_wrapper import VEmbedWrapper, _auto_detect_encoder_mode  # noqa: E402


class TestEncoderModeDetection:
    """Test encoder_mode auto-detection from model paths."""

    @pytest.mark.parametrize(
        "model_path,expected_mode",
        [
            # Qwen-VL (multimodal)
            ("experiments/output_qwen3_vl_embedding_2b", "qwen-vl"),
            ("Qwen/Qwen3-VL-Embedding-2B", "qwen-vl"),
            ("models/Qwen/Qwen3-VL-Embedding-8B", "qwen-vl"),
            ("checkpoint/qwen3_vl", "qwen-vl"),
            # Qwen-Embedding (text-only)
            ("experiments/output_qwen3_embedding", "qwen"),
            ("Qwen/Qwen3-Embedding-8B", "qwen"),
            ("checkpoint/qwen3_embedding", "qwen"),
            # SigLIP
            ("google/siglip-base-patch16-224", "siglip"),
            ("checkpoint/siglip", "siglip"),
            # Unknown (let auto-detect)
            ("openai/clip-vit-base-patch32", None),
            ("models/custom-model", None),
        ],
    )
    def test_auto_detect_encoder_mode(self, model_path, expected_mode):
        """Test encoder_mode auto-detection from various path formats."""
        result = _auto_detect_encoder_mode(model_path)
        assert result == expected_mode

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        assert _auto_detect_encoder_mode("QWEN3_VL_EMBEDDING") == "qwen-vl"
        assert _auto_detect_encoder_mode("QWEN3-EMBEDDING") == "qwen"
        assert _auto_detect_encoder_mode("SIGLIP") == "siglip"


class TestVEmbedWrapperInit:
    """Test VEmbedWrapper initialization with different configurations."""

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("vembed.model.processors.ProcessorRegistry")
    @patch("vembed.model.processors.registry.ProcessorRegistry")
    def test_init_with_encoder_mode(
        self, mock_proc_registry_local, mock_proc_registry, mock_model_cls
    ):
        """Test initialization with explicit encoder_mode."""
        mock_processor = MagicMock()
        # Create a mock loader class with a static load method
        mock_loader = MagicMock()
        mock_loader.load = MagicMock(return_value=mock_processor)

        mock_proc_registry.resolve.return_value = mock_processor
        mock_proc_registry.get.return_value = mock_loader
        mock_proc_registry_local.resolve.return_value = mock_processor
        mock_proc_registry_local.get.return_value = mock_loader

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        VEmbedWrapper("test_path", encoder_mode="qwen-vl")

        mock_proc_registry_local.get.assert_called_once_with("qwen-vl")
        mock_model_cls.assert_called_once()

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("vembed.model.processors.ProcessorRegistry")
    @patch("vembed.model.processors.registry.ProcessorRegistry")
    def test_init_with_attn_implementation(
        self, mock_proc_registry_local, mock_proc_registry, mock_model_cls
    ):
        """Test initialization with attention implementation hint."""
        mock_processor = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load = MagicMock(return_value=mock_processor)
        mock_proc_registry.resolve.return_value = mock_processor
        mock_proc_registry.get.return_value = mock_loader
        mock_proc_registry_local.resolve.return_value = mock_processor
        mock_proc_registry_local.get.return_value = mock_loader

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        VEmbedWrapper(
            "test_path",
            encoder_mode="qwen-vl",
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
        )

        mock_model_cls.assert_called_once()
        call_kwargs = mock_model_cls.call_args[1]
        assert call_kwargs["attn_implementation"] == "flash_attention_2"
        assert call_kwargs["torch_dtype"] == "bfloat16"


class TestVEmbedWrapperImageSupport:
    """Test supports_images property for different model types."""

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("vembed.model.processors.ProcessorRegistry")
    @patch("vembed.model.processors.registry.ProcessorRegistry")
    def test_multimodal_model_supports_images(
        self, mock_proc_registry_local, mock_proc_registry, mock_model_cls
    ):
        """Test that multimodal models report supports_images=True."""
        mock_processor = MagicMock()
        mock_processor.image_processor = MagicMock()
        mock_proc_registry.resolve.return_value = mock_processor
        mock_proc_registry.get.return_value = None  # No specific loader
        mock_proc_registry_local.resolve.return_value = mock_processor
        mock_proc_registry_local.get.return_value = None

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        wrapper = VEmbedWrapper("Qwen/Qwen3-VL-Embedding-2B")
        assert wrapper.supports_images is True

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("vembed.model.processors.ProcessorRegistry")
    @patch("vembed.model.processors.registry.ProcessorRegistry")
    def test_text_only_model_does_not_support_images(
        self, mock_proc_registry_local, mock_proc_registry, mock_model_cls
    ):
        """Test that text-only models report supports_images=False."""
        # Processor without image_processor indicates text-only model
        mock_processor = MagicMock()
        del mock_processor.image_processor
        mock_proc_registry.resolve.return_value = mock_processor
        mock_proc_registry.get.return_value = None  # No specific loader
        mock_proc_registry_local.resolve.return_value = mock_processor
        mock_proc_registry_local.get.return_value = None

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        wrapper = VEmbedWrapper("Qwen/Qwen3-Embedding-8B")
        assert wrapper.supports_images is False


class TestVEmbedWrapperEncoding:
    """Test encode_text and encode_image methods."""

    def test_encode_text_with_tokenizer(self):
        """Test text encoding with tokenizer-based processor."""
        with patch("benchmark.model_wrapper.VisualRetrievalModel") as mock_model_cls:
            # Setup mocks
            mock_processor = MagicMock()
            mock_processor.tokenizer = MagicMock()
            mock_processor.tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

            mock_model = MagicMock()
            mock_model.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_model.eval = MagicMock()
            mock_model_cls.return_value = mock_model

            wrapper = VEmbedWrapper.__new__(VEmbedWrapper)
            wrapper.processor = mock_processor
            wrapper.model = mock_model
            wrapper._supports_images = True

            # Encode
            embeddings = wrapper.encode_text(["test text"], device="cpu")

            assert embeddings.shape == (1, 3)
            mock_processor.tokenizer.assert_called_once()

    def test_encode_image_raises_for_text_only_model(self):
        """Test that encode_image raises for text-only models."""
        with patch("benchmark.model_wrapper.VisualRetrievalModel") as mock_model_cls:
            mock_processor = MagicMock()
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model_cls.return_value = mock_model

            wrapper = VEmbedWrapper.__new__(VEmbedWrapper)
            wrapper.processor = mock_processor
            wrapper.model = mock_model
            wrapper._supports_images = False  # Text-only model

            with pytest.raises(RuntimeError, match="does not support image encoding"):
                wrapper.encode_image([MagicMock()], device="cpu")


class TestTextOnlyWrapper:
    """Test TextOnlyWrapper for text-only models."""

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("benchmark.model_wrapper.ProcessorRegistry")
    def test_text_only_wrapper_init(self, mock_processor_registry, mock_model_cls):
        """Test TextOnlyWrapper initialization."""
        mock_processor = MagicMock()
        mock_processor_registry.resolve.return_value = mock_processor
        mock_processor_registry.get.return_value = None  # No specific loader

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        from benchmark.model_wrapper import TextOnlyWrapper

        wrapper = TextOnlyWrapper("Qwen/Qwen3-Embedding-8B")
        assert wrapper.supports_images is False

    @patch("benchmark.model_wrapper.VisualRetrievalModel")
    @patch("benchmark.model_wrapper.ProcessorRegistry")
    def test_text_only_wrapper_encode_image_raises(self, mock_processor_registry, mock_model_cls):
        """Test that TextOnlyWrapper.encode_image raises NotImplementedError."""
        mock_processor = MagicMock()
        mock_processor_registry.resolve.return_value = mock_processor
        mock_processor_registry.get.return_value = None  # No specific loader

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        from benchmark.model_wrapper import TextOnlyWrapper

        wrapper = TextOnlyWrapper("Qwen/Qwen3-Embedding-8B")

        with pytest.raises(NotImplementedError, match="text-only model"):
            wrapper.encode_image([MagicMock()], device="cpu")


def test_forward_interface_compatibility():
    """Test that wrapper uses unified forward() interface."""
    with patch("benchmark.model_wrapper.VisualRetrievalModel") as mock_model_cls:
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock that returns tuple
        mock_model = MagicMock()
        mock_model.return_value = (torch.tensor([[0.1, 0.2, 0.3]]), None)
        mock_model.eval = MagicMock()
        mock_model_cls.return_value = mock_model

        wrapper = VEmbedWrapper.__new__(VEmbedWrapper)
        wrapper.processor = mock_processor
        wrapper.model = mock_model
        wrapper._supports_images = True

        # Should handle tuple output
        embeddings = wrapper.encode_text(["test"], device="cpu")
        assert embeddings.shape == (1, 3)
