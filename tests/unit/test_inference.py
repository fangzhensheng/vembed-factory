"""Unit tests for vembed inference engine (VEmbedModel)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from vembed.inference import VEmbedModel


@pytest.fixture
def mock_model():
    """Create a mock VisualRetrievalModel."""
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock()
    # Mock forward pass - return embeddings of shape (batch_size, 768)
    model.return_value = torch.randn(4, 768, requires_grad=False)
    return model


@pytest.fixture
def mock_processor():
    """Create a mock unified processor."""
    processor = MagicMock()

    # Create a mock that returns a dict-like object with `.to()` method
    output_dict = MagicMock()
    output_dict.__getitem__ = lambda self, key: {
        "input_ids": torch.zeros(4, 10, dtype=torch.long),
        "attention_mask": torch.ones(4, 10, dtype=torch.long),
        "pixel_values": torch.randn(4, 3, 224, 224),
    }[key]
    output_dict.to = MagicMock(return_value=output_dict)

    processor.return_value = output_dict
    return processor


@pytest.fixture
def temp_model_path():
    """Create a temporary model directory with config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "vembed_config.json"
        config = {
            "pooling_method": "mean",
            "projection_dim": 768,
            "topk_tokens": 0,
            "use_mrl": False,
            "mrl_dims": None,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        yield tmpdir


class TestVEmbedModelInit:
    """Test VEmbedModel initialization."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_init_with_unified_processor(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test initialization with unified processor (non-composed mode)."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor

        model = VEmbedModel("openai/clip-vit-base-patch32", device="cpu")

        assert model.device == "cpu"
        assert model.encoder_mode == "auto"
        assert model.mrl_dim is None
        assert model.processor is not None
        assert model.text_processor is None
        assert model.image_processor is None
        mock_model.eval.assert_called_once()

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.build_text_processor")
    @patch("vembed.inference.build_image_processor")
    def test_init_with_composed_processors(
        self, mock_img_proc, mock_text_proc, mock_model_cls, mock_model
    ):
        """Test initialization with separate text/image processors (composed mode)."""
        mock_model_cls.return_value = mock_model
        mock_text_proc.return_value = MagicMock()
        mock_img_proc.return_value = MagicMock()

        model = VEmbedModel(
            "dummy/path",
            encoder_mode="composed",
            text_model_name="bert-base",
            image_model_name="clip",
            device="cpu",
        )

        assert model.encoder_mode == "composed"
        assert model.processor is None
        assert model.text_processor is not None
        assert model.image_processor is not None

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_init_with_vembed_config(
        self, mock_registry, mock_model_cls, mock_model, temp_model_path
    ):
        """Test loading vembed_config.json from checkpoint."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = MagicMock()

        model = VEmbedModel(temp_model_path, device="cpu")

        assert model.device == "cpu"
        mock_model_cls.assert_called_once()
        # Verify config was loaded
        call_kwargs = mock_model_cls.call_args[1]
        assert "projection_dim" in call_kwargs

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_init_with_mrl_dim(self, mock_registry, mock_model_cls, mock_model):
        """Test initialization with MRL dimension."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = MagicMock()

        model = VEmbedModel("dummy/path", device="cpu", mrl_dim=256)

        assert model.mrl_dim == 256


class TestVEmbedModelEncodeText:
    """Test text encoding functionality."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_single_text(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test encoding a single text string."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode_text("hello world", normalize=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_batch_text(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test encoding a batch of text strings."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(3, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        texts = ["hello", "world", "test"]
        embeddings = model.encode_text(texts, normalize=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_text_without_normalization(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test encoding text without L2 normalization."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(2, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode_text(["a", "b"], normalize=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_text_empty_string(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test encoding empty string."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode_text("", normalize=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)


class TestVEmbedModelEncodeImage:
    """Test image encoding functionality."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock PIL Image."""
        img = Image.new("RGB", (224, 224), color="red")
        return img

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_single_pil_image(
        self, mock_registry, mock_model_cls, mock_model, mock_processor, mock_image
    ):
        """Test encoding a single PIL Image."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode_image(mock_image, normalize=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_batch_pil_images(
        self, mock_registry, mock_model_cls, mock_model, mock_processor, mock_image
    ):
        """Test encoding a batch of PIL Images."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(3, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        images = [mock_image, mock_image, mock_image]
        embeddings = model.encode_image(images, normalize=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_image_from_file(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test encoding image from file path."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (224, 224), color="blue")
            img.save(img_path)

            model = VEmbedModel("dummy/path", device="cpu")
            embeddings = model.encode_image(str(img_path), normalize=True)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (1, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_image_without_normalization(
        self, mock_registry, mock_model_cls, mock_model, mock_processor, mock_image
    ):
        """Test encoding image without normalization."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode_image(mock_image, normalize=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)


class TestVEmbedModelGenericEncode:
    """Test generic encode interface."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_text_mode(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test generic encode for text (is_image=False)."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(2, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode(["hello", "world"], is_image=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_image_mode(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test generic encode for image (is_image=True)."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (224, 224))
            img.save(img_path)

            model = VEmbedModel("dummy/path", device="cpu")
            embeddings = model.encode(str(img_path), is_image=True)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (1, 768)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_encode_default_text_mode(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test generic encode defaults to text mode."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        mock_model.return_value = torch.randn(1, 768)

        model = VEmbedModel("dummy/path", device="cpu")
        embeddings = model.encode("hello")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 768)


class TestVEmbedModelMRLFeature:
    """Test Multi-Vector Retrieval (MRL) feature."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_truncate_embeddings_with_mrl(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test embedding truncation with MRL dimension."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        full_embeddings = torch.randn(2, 768)
        mock_model.return_value = full_embeddings

        model = VEmbedModel("dummy/path", device="cpu", mrl_dim=256)
        embeddings = model.encode_text(["a", "b"], normalize=True)

        assert embeddings.shape == (2, 256)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_no_truncation_without_mrl(
        self, mock_registry, mock_model_cls, mock_model, mock_processor
    ):
        """Test embeddings not truncated when MRL dimension is None."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        full_embeddings = torch.randn(1, 768)
        mock_model.return_value = full_embeddings

        model = VEmbedModel("dummy/path", device="cpu", mrl_dim=None)
        embeddings = model.encode_text("hello")

        assert embeddings.shape == (1, 768)


class TestVEmbedModelDeviceHandling:
    """Test device handling (CPU/GPU)."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_device_cpu(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test CPU device."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor

        model = VEmbedModel("dummy/path", device="cpu")
        assert model.device == "cpu"
        mock_model.to.assert_called_with("cpu")

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_device_cuda(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test CUDA device (mocked)."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor

        model = VEmbedModel("dummy/path", device="cuda:0")
        assert model.device == "cuda:0"
        mock_model.to.assert_called_with("cuda:0")


class TestVEmbedModelNormalization:
    """Test L2 normalization feature."""

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_normalize_embeddings(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test L2 normalization of embeddings."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        # Create embeddings with known norm
        embeddings = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)
        mock_model.return_value = embeddings

        model = VEmbedModel("dummy/path", device="cpu")
        normalized = model.encode_text(["a", "b"], normalize=True)

        # Verify L2 norm is ~1.0
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=6)

    @patch("vembed.inference.VisualRetrievalModel")
    @patch("vembed.inference.ProcessorRegistry")
    def test_skip_normalization(self, mock_registry, mock_model_cls, mock_model, mock_processor):
        """Test skipping normalization."""
        mock_model_cls.return_value = mock_model
        mock_registry.resolve.return_value = mock_processor
        embeddings = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        mock_model.return_value = embeddings

        model = VEmbedModel("dummy/path", device="cpu")
        unnormalized = model.encode_text("hello", normalize=False)

        # Verify norm != 1.0
        norm = np.linalg.norm(unnormalized)
        assert not np.isclose(norm, 1.0)


class TestVEmbedModelStaticMethods:
    """Test static processor loading methods."""

    @patch("vembed.inference.AutoTokenizer")
    def test_load_text_processor_from_checkpoint(self, mock_tokenizer):
        """Test loading tokenizer from checkpoint."""
        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        result = VEmbedModel._load_text_processor("/path/to/checkpoint", None)

        assert result == mock_tok
        mock_tokenizer.from_pretrained.assert_called_once_with("/path/to/checkpoint")

    @patch("vembed.inference.build_text_processor")
    @patch("vembed.inference.AutoTokenizer")
    def test_load_text_processor_fallback(self, mock_tokenizer, mock_build):
        """Test fallback when checkpoint tokenizer not found."""
        mock_tokenizer.from_pretrained.side_effect = OSError("Not found")
        mock_tok = MagicMock()
        mock_build.return_value = mock_tok

        result = VEmbedModel._load_text_processor("/path/to/checkpoint", "bert-base")

        assert result == mock_tok
        mock_build.assert_called_once_with("bert-base")

    @patch("vembed.inference.AutoTokenizer")
    def test_load_text_processor_no_fallback_raises(self, mock_tokenizer):
        """Test error when no fallback provided."""
        mock_tokenizer.from_pretrained.side_effect = OSError("Not found")

        with pytest.raises(ValueError):
            VEmbedModel._load_text_processor("/path/to/checkpoint", None)

    @patch("vembed.inference.AutoImageProcessor")
    def test_load_image_processor_from_checkpoint(self, mock_image_proc):
        """Test loading image processor from checkpoint."""
        mock_proc = MagicMock()
        mock_image_proc.from_pretrained.return_value = mock_proc

        result = VEmbedModel._load_image_processor("/path/to/checkpoint", None)

        assert result == mock_proc
        mock_image_proc.from_pretrained.assert_called_once_with("/path/to/checkpoint")

    @patch("vembed.inference.build_image_processor")
    @patch("vembed.inference.AutoImageProcessor")
    def test_load_image_processor_fallback(self, mock_image_proc, mock_build):
        """Test fallback when checkpoint image processor not found."""
        mock_image_proc.from_pretrained.side_effect = OSError("Not found")
        mock_proc = MagicMock()
        mock_build.return_value = mock_proc

        result = VEmbedModel._load_image_processor("/path/to/checkpoint", "clip")

        assert result == mock_proc
        mock_build.assert_called_once_with("clip")

    @patch("vembed.inference.ProcessorRegistry")
    def test_load_unified_processor(self, mock_registry):
        """Test loading unified processor via registry."""
        mock_proc = MagicMock()
        mock_registry.resolve.return_value = mock_proc

        result = VEmbedModel._load_unified_processor("/path/to/model")

        assert result == mock_proc
        mock_registry.resolve.assert_called_once_with("/path/to/model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
