"""Tests for Qwen3 model registration and basic functionality.

Tests cover:
- qwen3_vl model registration (multimodal)
- qwen3_embedding model registration (text-only)
- Processor registration
- Basic forward pass functionality
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vembed.model.registry import ModelRegistry
from vembed.model.processors.registry import ProcessorRegistry


class TestQwen3ModelRegistration:
    """Test that Qwen3 models are properly registered."""

    def test_qwen3_vl_model_registered(self):
        """Test that qwen3_vl model is registered."""
        assert "qwen3_vl" in ModelRegistry.list_models()
        cls = ModelRegistry.get("qwen3_vl")
        assert cls.__name__ == "Qwen3VLEmbeddingModel"

    def test_qwen3_embedding_model_registered(self):
        """Test that qwen3_embedding model is registered."""
        assert "qwen3_embedding" in ModelRegistry.list_models()
        cls = ModelRegistry.get("qwen3_embedding")
        assert cls.__name__ == "Qwen3EmbeddingTextModel"

    def test_qwen3_vl_processor_registered(self):
        """Test that qwen3_vl processor is registered."""
        assert "qwen3_vl" in ProcessorRegistry.list_loaders()

    def test_qwen3_embedding_processor_registered(self):
        """Test that qwen3_embedding processor is registered."""
        assert "qwen3_embedding" in ProcessorRegistry.list_loaders()


class TestQwen3VLEmbeddingModel:
    """Test Qwen3VLEmbeddingModel (multimodal) functionality."""

    @patch("vembed.model.backbones.qwen3_vl_embedding.AutoModel")
    @patch("vembed.model.backbones.qwen3_vl_embedding.AutoConfig")
    def test_init_basic(self, mock_config, mock_auto_model):
        """Test basic initialization."""
        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 3584
        mock_cfg.text_config = None
        mock_config.from_pretrained.return_value = mock_cfg

        mock_model = MagicMock()
        mock_model.config = mock_cfg
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.backbones.qwen3_vl_embedding import Qwen3VLEmbeddingModel

        config = {"model_name_or_path": "Qwen/Qwen3-VL-Embedding-2B"}
        model = Qwen3VLEmbeddingModel(config)

        assert model.feature_dim == 3584
        mock_auto_model.from_pretrained.assert_called_once()

    @patch("vembed.model.backbones.qwen3_vl_embedding.AutoModel")
    def test_forward_with_vision_inputs(self, mock_auto_model):
        """Test forward pass with vision inputs."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.text_config = None
        mock_model.config.hidden_size = 3584

        # Mock outputs
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 3584)
        mock_model.return_value = mock_output
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.backbones.qwen3_vl_embedding import Qwen3VLEmbeddingModel

        config = {"model_name_or_path": "Qwen/Qwen3-VL-Embedding-2B"}
        model = Qwen3VLEmbeddingModel(config)

        # Forward with vision inputs
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        pixel_values = torch.randn(1, 3, 224, 224)
        image_grid_thw = torch.tensor([[1, 2, 2]])

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        assert output.shape == (1, 3584)  # Normalized embeddings
        # Check L2 normalization
        assert torch.allclose(output.norm(dim=1), torch.ones(1), atol=1e-5)

    @patch("vembed.model.backbones.qwen3_vl_embedding.AutoModel")
    def test_pool_last_token(self, mock_auto_model):
        """Test last token pooling implementation."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 3584
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.backbones.qwen3_vl_embedding import Qwen3VLEmbeddingModel

        # Create test data
        hidden = torch.randn(2, 10, 3584)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5 tokens
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 8 tokens
        ])

        result = Qwen3VLEmbeddingModel._pool_last_token(hidden, attention_mask)

        # Should extract the last non-padding token for each sequence
        assert result.shape == (2, 3584)
        # First sequence: index 4 (5th token, 0-indexed)
        assert torch.allclose(result[0], hidden[0, 4])
        # Second sequence: index 7 (8th token)
        assert torch.allclose(result[1], hidden[1, 7])


class TestQwen3EmbeddingTextModel:
    """Test Qwen3EmbeddingTextModel (text-only) functionality."""

    @patch("vembed.model.backbones.qwen3_embedding.AutoModel")
    @patch("vembed.model.backbones.qwen3_embedding.AutoConfig")
    def test_init_basic(self, mock_config, mock_auto_model):
        """Test basic initialization."""
        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 3584
        mock_config.from_pretrained.return_value = mock_cfg

        mock_model = MagicMock()
        mock_model.config = mock_cfg
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.backbones.qwen3_embedding import Qwen3EmbeddingTextModel

        config = {"model_name_or_path": "Qwen/Qwen3-Embedding-8B"}
        model = Qwen3EmbeddingTextModel(config)

        assert model.feature_dim == 3584

    @patch("vembed.model.backbones.qwen3_embedding.AutoModel")
    def test_forward_text_only(self, mock_auto_model):
        """Test forward pass without vision inputs."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 3584

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 3584)
        mock_model.return_value = mock_output
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.backbones.qwen3_embedding import Qwen3EmbeddingTextModel

        config = {"model_name_or_path": "Qwen/Qwen3-Embedding-8B"}
        model = Qwen3EmbeddingTextModel(config)

        # Forward with text only (no pixel_values)
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.shape == (1, 3584)
        # Check L2 normalization
        assert torch.allclose(output.norm(dim=1), torch.ones(1), atol=1e-5)


class TestQwen3Processors:
    """Test Qwen3 processor loaders."""

    @patch("vembed.model.processors.qwen3_vl.Qwen3VLProcessorLoader.load")
    def test_qwen3_vl_processor_loading(self, mock_load):
        """Test Qwen3-VL processor loading."""
        mock_load.return_value = MagicMock()

        from vembed.model.processors.qwen3_vl import Qwen3VLProcessorLoader

        # Test match function
        assert Qwen3VLProcessorLoader.match("Qwen/Qwen3-VL-Embedding-2B")
        assert Qwen3VLProcessorLoader.match("qwen3_vl_embedding")
        assert not Qwen3VLProcessorLoader.match("qwen3_embedding")

    @patch("transformers.AutoTokenizer")
    def test_qwen3_embedding_processor_loading(self, mock_tokenizer):
        """Test Qwen3-Embedding processor (tokenizer) loading."""
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer.from_pretrained = MagicMock(return_value=MagicMock())
        mock_tokenizer.AutoTokenizer = mock_tokenizer_cls

        from vembed.model.processors.qwen3_embedding import Qwen3EmbeddingProcessorLoader

        # Test match function
        assert Qwen3EmbeddingProcessorLoader.match("Qwen/Qwen3-Embedding-8B")
        assert Qwen3EmbeddingProcessorLoader.match("qwen3_embedding")
        assert not Qwen3EmbeddingProcessorLoader.match("qwen3-vl")

        # Test loading (verify padding_side is set to "left")
        Qwen3EmbeddingProcessorLoader.load("test/path")

        # The actual tokenizer call should have padding_side="left"
        # This is verified by the implementation, but we can't easily test
        # without a real tokenizer


class TestModelIntegration:
    """Integration tests for Qwen3 models through VisualRetrievalModel."""

    @patch("vembed.model.backbones.qwen3_vl_embedding.AutoModel")
    def test_vl_model_through_visual_retrieval_model(self, mock_auto_model):
        """Test that qwen3_vl mode works through VisualRetrievalModel."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.text_config = None
        mock_model.config.hidden_size = 3584

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 3584)
        mock_model.return_value = mock_output
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.modeling import VisualRetrievalModel

        model = VisualRetrievalModel(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            encoder_mode="qwen3_vl",
        )

        # Verify backend is set correctly
        from vembed.model.backbones.qwen3_vl_embedding import Qwen3VLEmbeddingModel
        assert isinstance(model.backend, Qwen3VLEmbeddingModel)

    @patch("vembed.model.backbones.qwen3_embedding.AutoModel")
    def test_embedding_model_through_visual_retrieval_model(self, mock_auto_model):
        """Test that qwen3_embedding mode works through VisualRetrievalModel."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 3584

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 3584)
        mock_model.return_value = mock_output
        mock_auto_model.from_pretrained.return_value = mock_model

        from vembed.model.modeling import VisualRetrievalModel

        model = VisualRetrievalModel(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            encoder_mode="qwen3_embedding",
        )

        # Verify backend is set correctly
        from vembed.model.backbones.qwen3_embedding import Qwen3EmbeddingTextModel
        assert isinstance(model.backend, Qwen3EmbeddingTextModel)
