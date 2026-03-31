"""Configuration manager for dataset_info.json and training_info.json."""

import json
import os
from pathlib import Path
from typing import Any


class ConfigManager:
    """Manages dataset and training configurations from JSON files."""

    def __init__(self, examples_dir: str | None = None):
        """Initialize config manager.

        Args:
            examples_dir: Path to examples directory. If None, uses current directory.
        """
        if examples_dir is None:
            examples_dir = os.path.dirname(__file__)
        self.examples_dir = Path(examples_dir)
        self._dataset_info: dict[str, Any] = {}
        self._training_info: dict[str, Any] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load dataset_info.json and training_info.json."""
        dataset_file = self.examples_dir / "dataset_info.json"
        training_file = self.examples_dir / "training_info.json"

        if dataset_file.exists():
            with open(dataset_file) as f:
                self._dataset_info = json.load(f)

        if training_file.exists():
            with open(training_file) as f:
                self._training_info = json.load(f)

    def get_dataset(self, dataset_name: str) -> dict[str, Any] | None:
        """Get dataset configuration by name."""
        return self._dataset_info.get(dataset_name)

    def get_training_config(self, config_name: str) -> dict[str, Any] | None:
        """Get training configuration by name."""
        return self._training_info.get(config_name)

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        return list(self._dataset_info.keys())

    def list_training_configs(self) -> list[str]:
        """List all available training configurations."""
        return list(self._training_info.keys())

    def resolve_dataset_paths(self, dataset_name: str) -> dict[str, str]:
        """Resolve actual file paths for a dataset.

        Returns:
            Dict with 'data_path', 'image_root', 'val_data_path' keys.
        """
        dataset_info = self.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Map dataset_name to directory convention
        dataset_dir_map = {
            "msmarco_t2t": "msmarco",
            "flickr30k_t2i": "flickr30k",
            "stanford_online_products_i2i": "stanford_online_products",
            "coco_t2i": "coco",
        }

        dataset_dir = dataset_dir_map.get(dataset_name, dataset_name)

        # Try multiple locations: data/, examples/data/
        possible_roots = [
            Path("data") / dataset_dir,
            self.examples_dir / "data" / dataset_dir,
            Path(dataset_name),
        ]

        data_root = None
        for root in possible_roots:
            if root.exists():
                data_root = root
                break

        if not data_root:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' directory not found in {[str(r) for r in possible_roots]}"
            )

        return {
            "data_path": str(data_root / "train.jsonl"),
            "val_data_path": str(data_root / "val.jsonl") if (data_root / "val.jsonl").exists() else "",
            "image_root": str(data_root),
        }

    def get_training_command(self, config_name: str) -> dict[str, Any]:
        """Get complete training command information.

        Returns:
            Dict with config_path, dataset_paths, and training arguments.
        """
        training_cfg = self.get_training_config(config_name)
        if not training_cfg:
            raise ValueError(f"Training config '{config_name}' not found")

        dataset_name = training_cfg.get("dataset")
        dataset_paths = self.resolve_dataset_paths(dataset_name) if dataset_name else {}

        return {
            "config_name": config_name,
            "description": training_cfg.get("description", ""),
            "dataset": dataset_name,
            "dataset_paths": dataset_paths,
            "training_config": training_cfg,
        }


if __name__ == "__main__":
    # Demo usage
    manager = ConfigManager()

    print("Available datasets:")
    for ds in manager.list_datasets():
        print(f"  - {ds}")

    print("\nAvailable training configs:")
    for cfg in manager.list_training_configs():
        info = manager.get_training_config(cfg)
        print(f"  - {cfg}: {info.get('description', '')}")

    # Try to resolve a training command
    try:
        cmd_info = manager.get_training_command("clip_t2i")
        print(f"\nTraining command for 'clip_t2i':")
        print(f"  Description: {cmd_info['description']}")
        print(f"  Dataset: {cmd_info['dataset']}")
        print(f"  Dataset paths: {cmd_info['dataset_paths']}")
    except Exception as e:
        print(f"\nError: {e}")
