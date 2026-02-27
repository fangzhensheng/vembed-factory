"""Unit tests for data pipeline components."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.training.model_builder import load_processor


@pytest.fixture
def temp_jsonl_data():
    """Create temporary JSONL dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        data = [
            {"query": "a cat", "positive": "cat.jpg", "negatives": ["dog.jpg"]},
            {"query": "a dog", "positive": "dog.jpg", "negatives": ["cat.jpg"]},
            {"query": "a bird", "positive": "bird.jpg", "negatives": ["cat.jpg"]},
        ]

        # Write JSONL
        data_path = tmpdir / "data.jsonl"
        with open(data_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        # Create image directory
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        for img_name in ["cat.jpg", "dog.jpg", "bird.jpg"]:
            (img_dir / img_name).touch()

        yield tmpdir


class TestDatasetCreation:
    """Test dataset creation."""

    def test_create_dataset(self, temp_jsonl_data):
        """Test creating a dataset."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        assert dataset is not None
        assert len(dataset) == 3

    def test_dataset_getitem(self, temp_jsonl_data):
        """Test getting items from dataset."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        sample = dataset[0]

        # Should have text and image data
        assert sample is not None
        assert isinstance(sample, dict)

    def test_dataset_length(self, temp_jsonl_data):
        """Test dataset length."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        assert len(dataset) == 3


class TestCollatorRegistry:
    """Test collator registry."""

    def test_default_collator_exists(self):
        """Test that default collator is registered."""
        collator = CollatorRegistry.get("default")

        assert collator is not None

    def test_create_default_collator(self):
        """Test creating default collator."""
        processor = load_processor("openai/clip-vit-base-patch32")

        collator_cls = CollatorRegistry.get("default")
        collator = collator_cls(processor=processor, mode="train")

        assert collator is not None
        assert callable(collator)

    def test_collator_registry_names(self):
        """Test available collator names."""
        names = CollatorRegistry.list_names()

        assert isinstance(names, (list, tuple, set))
        assert "default" in names or len(names) > 0


class TestDataLoader:
    """Test DataLoader creation and usage."""

    def test_create_dataloader(self, temp_jsonl_data):
        """Test creating a DataLoader."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        collator_cls = CollatorRegistry.get("default")
        collator = collator_cls(processor=processor, mode="train")

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=True,
        )

        assert dataloader is not None
        assert len(dataloader) > 0

    def test_dataloader_iteration(self, temp_jsonl_data):
        """Test iterating through DataLoader."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        collator_cls = CollatorRegistry.get("default")
        collator = collator_cls(processor=processor, mode="train")

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
        )

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert batch is not None
            assert isinstance(batch, dict)

        assert batch_count > 0

    def test_batch_size(self, temp_jsonl_data):
        """Test batch size is respected."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        collator_cls = CollatorRegistry.get("default")
        collator = collator_cls(processor=processor, mode="train")

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
        )

        batch = next(iter(dataloader))

        # Check batch size
        batch_size = next(iter(batch.values())).shape[0]
        assert batch_size <= 2


class TestDatasetModes:
    """Test different dataset modes."""

    def test_train_mode(self, temp_jsonl_data):
        """Test dataset in train mode."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="train",
        )

        assert dataset is not None

    def test_eval_mode(self, temp_jsonl_data):
        """Test dataset in eval mode."""
        processor = load_processor("openai/clip-vit-base-patch32")

        dataset = VisualRetrievalDataset(
            data_source=str(temp_jsonl_data / "data.jsonl"),
            processor=processor,
            image_root=str(temp_jsonl_data / "images"),
            mode="eval",
        )

        assert dataset is not None


class TestCollatorTypes:
    """Test different collator types."""

    def test_get_available_collators(self):
        """Test getting available collators."""
        collators = CollatorRegistry.list_names()

        assert len(collators) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
