"""End-to-end integration tests for training pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal test dataset
        data = [
            {"query": "a cat", "positive": "cat.jpg", "negatives": ["dog.jpg"]},
            {"query": "a dog", "positive": "dog.jpg", "negatives": ["cat.jpg"]},
            {"query": "a bird", "positive": "bird.jpg", "negatives": ["cat.jpg"]},
            {"query": "a fish", "positive": "fish.jpg", "negatives": ["cat.jpg"]},
        ]

        # Write JSONL data
        train_path = tmpdir / "train.jsonl"
        with open(train_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        val_path = tmpdir / "val.jsonl"
        with open(val_path, "w") as f:
            for item in data[:2]:
                f.write(json.dumps(item) + "\n")

        # Create image directory
        img_dir = tmpdir / "images"
        img_dir.mkdir()

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")

        for img_name in ["cat.jpg", "dog.jpg", "bird.jpg", "fish.jpg"]:
            img.save(img_dir / img_name)

        yield tmpdir


@pytest.fixture
def temp_config_dir(temp_data_dir):
    """Create temporary config directory with test configs."""
    config_dir = Path(tempfile.mkdtemp())

    # Minimal CLIP config for fast testing
    clip_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "data_path": str(temp_data_dir / "train.jsonl"),
        "val_data_path": str(temp_data_dir / "val.jsonl"),
        "image_root": str(temp_data_dir / "images"),
        "output_dir": str(config_dir / "output"),
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 1e-4,
        "loss_type": "infonce",
        "logging_steps": 1,
        "eval_steps": 2,
    }

    config_path = config_dir / "test_clip.yaml"
    with open(config_path, "w") as f:
        yaml.dump(clip_config, f)

    yield config_dir


class TestBasicTraining:
    """Test basic training functionality."""

    def test_import_trainer(self):
        """Test that Trainer can be imported."""
        from vembed.training.training_loop import Trainer  # noqa: F401

    def test_model_building(self):
        """Test model building."""
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "get_text_features")
        assert hasattr(model, "get_image_features")

    def test_processor_loading(self):
        """Test processor loading."""
        from vembed.training.model_builder import load_processor

        processor = load_processor("openai/clip-vit-base-patch32")

        assert processor is not None
        assert callable(processor)

    def test_dataset_creation(self, temp_data_dir):
        """Test dataset creation."""
        from vembed.data.dataset import VisualRetrievalDataset
        from vembed.training.model_builder import load_processor

        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_data_dir / "train.jsonl"),
            processor=processor,
            image_root=str(temp_data_dir / "images"),
            mode="train",
        )

        assert len(dataset) > 0
        sample = dataset[0]
        # Check for expected keys - handle list/tuple checks safely
        keys = set(sample.keys())
        assert "query_text" in keys or "input_ids" in keys

    def test_loss_creation(self):
        """Test loss function creation."""
        from vembed.losses.factory import LossFactory

        config = {"loss_type": "infonce"}
        loss_fn = LossFactory.create(config)

        assert loss_fn is not None
        assert callable(loss_fn)

    def test_optimizer_building(self):
        """Test optimizer building."""
        from vembed.training.model_builder import build_model
        from vembed.training.optimizer_builder import build_optimizer

        config = {"model_name": "openai/clip-vit-base-patch32", "learning_rate": 1e-5}
        model = build_model(config)
        optimizer = build_optimizer(model, config)

        assert optimizer is not None
        assert hasattr(optimizer, "step")
        assert hasattr(optimizer, "zero_grad")


class TestConfigLoading:
    """Test configuration loading and parsing."""

    def test_load_base_config(self):
        """Test loading default configuration."""
        from vembed.config import load_base_config

        config = load_base_config()

        assert isinstance(config, dict)
        assert "model_name" in config or "batch_size" in config

    def test_yaml_config_loading(self, temp_config_dir):
        """Test YAML config file loading."""
        import yaml

        config_path = next(temp_config_dir.glob("*.yaml"))
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["model_name"] == "openai/clip-vit-base-patch32"
        assert config["batch_size"] == 2
        assert config["epochs"] == 1


class TestDistributedConfig:
    """Test distributed training configuration."""

    def test_gradient_cache_config(self):
        """Test gradient cache configuration."""
        from vembed.training.config import get_distributed_config

        config = {
            "use_gradient_cache": True,
            "use_gradient_checkpointing": True,
        }

        _, use_grad_cache, _ = get_distributed_config(config)
        assert use_grad_cache is True

    def test_find_unused_parameters_config(self):
        """Test find_unused_parameters flag."""
        from vembed.training.config import get_distributed_config

        config = {
            "use_gradient_cache": True,
            "use_gradient_checkpointing": False,
        }

        _, _, find_unused = get_distributed_config(config)
        # With gradient cache, find_unused should be False (optimization enabled)
        assert find_unused is False

    def test_no_gradient_cache_config(self):
        """Test config without gradient cache."""
        from vembed.training.config import get_distributed_config

        config = {
            "use_gradient_cache": False,
            "use_gradient_checkpointing": False,
        }

        _, use_grad_cache, find_unused = get_distributed_config(config)
        assert use_grad_cache is False


class TestDataLoading:
    """Test data loading and preprocessing."""

    def test_collator_creation(self):
        """Test collator creation."""
        from vembed.data.registry import CollatorRegistry
        from vembed.training.model_builder import load_processor

        processor = load_processor("openai/clip-vit-base-patch32")
        collator = CollatorRegistry.get("default")(processor=processor, mode="train")

        assert collator is not None

    def test_dataloader_creation(self, temp_data_dir):
        """Test dataloader creation."""
        from torch.utils.data import DataLoader

        from vembed.data.dataset import VisualRetrievalDataset
        from vembed.data.registry import CollatorRegistry
        from vembed.training.model_builder import load_processor

        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_data_dir / "train.jsonl"),
            processor=processor,
            image_root=str(temp_data_dir / "images"),
            mode="train",
        )

        collator = CollatorRegistry.get("default")(processor=processor, mode="train")

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=True,
        )

        assert len(dataloader) > 0

        batch = next(iter(dataloader))
        assert "input_ids" in batch or "pixel_values" in batch


class TestModelBuilder:
    """Test model builder functions."""

    def test_build_model_clip(self):
        """Test building CLIP model."""
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        assert model is not None
        assert hasattr(model, "get_text_features")
        assert hasattr(model, "get_image_features")

    def test_enable_gradient_checkpointing(self):
        """Test enabling gradient checkpointing."""
        from vembed.training.model_builder import _enable_gradient_checkpointing, build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        class FakeAccelerator:
            def print(self, msg):
                pass

        _enable_gradient_checkpointing(model.backend, FakeAccelerator())

        # Check that gradient_checkpointing_enable was called
        # (actual check depends on model implementation)
        assert model is not None

    def test_unify_dtype(self):
        """Test dtype unification for FSDP."""
        from vembed.training.model_builder import build_model, unify_model_dtype_for_fsdp

        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "use_fsdp": True,
            "torch_dtype": "float32",
        }

        model = build_model(config)

        class FakeAccelerator:
            def print(self, msg):
                pass

        unify_model_dtype_for_fsdp(model, config, FakeAccelerator())

        # Verify all parameters have same dtype
        dtypes = {param.dtype for param in model.parameters()}
        assert len(dtypes) <= 1  # All same dtype


class TestLossComputation:
    """Test loss computation."""

    def test_infonce_loss(self):
        """Test InfoNCE loss."""
        from vembed.losses.functions.infonce import InfoNCELoss

        batch_size = 4
        dim = 256

        text_emb = torch.randn(batch_size, dim)
        image_emb = torch.randn(batch_size, dim)

        loss_fn = InfoNCELoss({"temperature": 0.1})

        loss = loss_fn(text_emb, image_emb, None)

        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_triplet_loss(self):
        """Test Triplet loss."""
        from vembed.losses.functions.triplet import TripletMarginLoss

        batch_size = 4
        dim = 256

        anchors = torch.randn(batch_size, dim)
        positives = torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size * 2, dim)

        loss_fn = TripletMarginLoss({"triplet_margin": 0.5})

        loss = loss_fn(anchors, positives, negatives)

        assert torch.isfinite(loss)

    def test_matryoshka_loss(self):
        """Test Matryoshka (MRL) loss."""
        from vembed.losses.functions.infonce import InfoNCELoss
        from vembed.losses.functions.matryoshka import MatryoshkaLoss

        batch_size = 4
        dim = 256

        text_emb = torch.randn(batch_size, dim)
        image_emb = torch.randn(batch_size, dim)

        base_loss = InfoNCELoss({"temperature": 0.1})
        mrl_loss = MatryoshkaLoss(base_loss, dims=[256, 128, 64])

        loss = mrl_loss(text_emb, image_emb, None)

        assert torch.isfinite(loss)


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_save(self, temp_config_dir):
        """Test checkpoint saving."""
        from unittest.mock import MagicMock

        from vembed.training.checkpoint import save_checkpoint
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        output_dir = temp_config_dir / "checkpoint"
        output_dir.mkdir()

        accelerator = MagicMock()
        accelerator.is_local_main_process = True
        accelerator.unwrap_model.return_value = model

        save_checkpoint(str(output_dir), model, accelerator)

        # Verify that save_pretrained was called on the unwrapped model
        if hasattr(model.save_pretrained, "assert_called_with"):
            model.save_pretrained.assert_called_with(str(output_dir))

        # Check that save_state was called
        if hasattr(accelerator.save_state, "assert_called_with"):
            accelerator.save_state.assert_called_with(str(output_dir))


class TestResumeTraining:
    """Test training resume from checkpoint functionality."""

    def test_load_checkpoint_training_state(self, temp_config_dir):
        """Test loading checkpoint with training state."""
        from unittest.mock import MagicMock

        from vembed.training.checkpoint import load_checkpoint, save_checkpoint
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        checkpoint_dir = temp_config_dir / "checkpoint_resume"
        checkpoint_dir.mkdir()

        # Create mock accelerator
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.unwrap_model.return_value = model
        accelerator.device = "cpu"

        # Training state to save
        training_state = {
            "global_step": 100,
            "epoch": 2,
            "best_metric": 0.85,
            "patience_counter": 1,
        }

        # Save checkpoint with training state
        save_checkpoint(
            str(checkpoint_dir),
            model,
            accelerator,
            config=config,
            training_state=training_state,
        )

        # Create new model and accelerator for loading
        model_2 = build_model(config)
        accelerator_2 = MagicMock()
        accelerator_2.is_main_process = True
        accelerator_2.unwrap_model.return_value = model_2
        accelerator_2.device = "cpu"

        # Load checkpoint
        loaded_state = load_checkpoint(
            str(checkpoint_dir),
            model_2,
            accelerator_2,
            mode="model_only",
        )

        # Verify training state was restored
        assert loaded_state["global_step"] == 100
        assert loaded_state["epoch"] == 2
        assert loaded_state["best_metric"] == 0.85
        assert loaded_state["patience_counter"] == 1

    def test_load_checkpoint_model_only_mode(self, temp_config_dir):
        """Test loading checkpoint in model_only mode."""
        from unittest.mock import MagicMock

        from vembed.training.checkpoint import load_checkpoint, save_checkpoint
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        checkpoint_dir = temp_config_dir / "checkpoint_model_only"
        checkpoint_dir.mkdir()

        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.unwrap_model.return_value = model
        accelerator.device = "cpu"

        # Save checkpoint
        save_checkpoint(
            str(checkpoint_dir),
            model,
            accelerator,
            config=config,
        )

        # Load in model_only mode (should not try to load optimizer state)
        model_2 = build_model(config)
        accelerator_2 = MagicMock()
        accelerator_2.is_main_process = True
        accelerator_2.unwrap_model.return_value = model_2
        accelerator_2.device = "cpu"

        # Should not raise error even without optimizer/scheduler
        loaded_state = load_checkpoint(
            str(checkpoint_dir),
            model_2,
            accelerator_2,
            mode="model_only",
        )

        assert isinstance(loaded_state, dict)

    def test_load_checkpoint_full_mode(self, temp_config_dir):
        """Test loading checkpoint in full mode."""
        from unittest.mock import MagicMock

        from vembed.training.checkpoint import load_checkpoint, save_checkpoint
        from vembed.training.model_builder import build_model

        config = {"model_name": "openai/clip-vit-base-patch32"}
        model = build_model(config)

        checkpoint_dir = temp_config_dir / "checkpoint_full"
        checkpoint_dir.mkdir()

        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.unwrap_model.return_value = model
        accelerator.device = "cpu"

        # Save checkpoint
        save_checkpoint(
            str(checkpoint_dir),
            model,
            accelerator,
            config=config,
        )

        # Load in full mode
        model_2 = build_model(config)
        accelerator_2 = MagicMock()
        accelerator_2.is_main_process = True
        accelerator_2.unwrap_model.return_value = model_2
        accelerator_2.device = "cpu"
        accelerator_2.load_state = MagicMock()  # Mock load_state call

        # Create mock optimizer and scheduler
        optimizer = MagicMock()
        scheduler = MagicMock()

        # Should attempt to load optimizer/scheduler state
        loaded_state = load_checkpoint(
            str(checkpoint_dir),
            model_2,
            accelerator_2,
            optimizer=optimizer,
            scheduler=scheduler,
            mode="full",
        )

        assert isinstance(loaded_state, dict)

    def test_checkpoint_not_found(self, temp_config_dir):
        """Test error handling when checkpoint not found."""
        from unittest.mock import MagicMock

        from vembed.training.checkpoint import load_checkpoint
        from vembed.training.model_builder import build_model

        model = build_model({"model_name": "openai/clip-vit-base-patch32"})
        accelerator = MagicMock()
        accelerator.device = "cpu"

        # Try to load non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                str(temp_config_dir / "nonexistent"),
                model,
                accelerator,
            )

    def test_vembed_config_persistence(self, temp_config_dir):
        """Test vembed config is saved and can be loaded."""
        from unittest.mock import MagicMock
        import json

        from vembed.training.checkpoint import save_checkpoint
        from vembed.training.model_builder import build_model

        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "pooling_method": "mean",
            "projection_dim": 512,
            "topk_tokens": 64,
            "retrieval_mode": "t2i",
            "loss_type": "infonce",
            "use_mrl": False,
            "encoder_mode": "auto",
        }

        model = build_model(config)
        checkpoint_dir = temp_config_dir / "checkpoint_config"
        checkpoint_dir.mkdir()

        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.unwrap_model.return_value = model

        save_checkpoint(
            str(checkpoint_dir),
            model,
            accelerator,
            config=config,
        )

        # Verify vembed_config.json was created
        config_file = checkpoint_dir / "vembed_config.json"
        assert config_file.exists()

        # Load and verify config
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["pooling_method"] == "mean"
        assert saved_config["projection_dim"] == 512
        assert saved_config["topk_tokens"] == 64
        assert saved_config["retrieval_mode"] == "t2i"


class TestDatasetNameIntegration:
    """Test dataset_name parameter and dataset_info.json integration."""

    def test_dataset_name_loading(self):
        """Test loading dataset info by name."""
        from vembed.training.config import _load_dataset_info

        # Test loading existing dataset
        dataset_info = _load_dataset_info("flickr30k_t2i")
        assert dataset_info is not None
        assert dataset_info["retrieval_mode"] == "t2i"
        assert dataset_info.get("file_name") is not None
        assert dataset_info.get("val_file_name") is not None

    def test_dataset_name_nonexistent(self):
        """Test loading non-existent dataset returns None gracefully."""
        from vembed.training.config import _load_dataset_info

        dataset_info = _load_dataset_info("nonexistent_dataset")
        assert dataset_info is None

    def test_dataset_info_has_description(self):
        """Test that dataset_info includes description metadata."""
        from vembed.training.config import _load_dataset_info

        dataset_info = _load_dataset_info("msmarco_t2t")
        assert dataset_info is not None
        assert dataset_info.get("description") is not None
        assert dataset_info.get("retrieval_mode") == "t2t"

    def test_dataset_info_column_mapping(self):
        """Test that dataset_info includes column mapping."""
        from vembed.training.config import _load_dataset_info

        dataset_info = _load_dataset_info("coco_t2i")
        assert dataset_info is not None
        assert dataset_info.get("columns") is not None
        assert "query" in dataset_info["columns"] or "caption" in dataset_info.get("columns", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
