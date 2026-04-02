"""Dataset configuration manager for vembed-factory.

Manages dataset_info.json to provide centralized dataset registry and path resolution.
Training configurations are managed directly via YAML files.
"""

import json
from pathlib import Path
from typing import Any


class DatasetManager:
    """Manages dataset configurations from dataset_info.json."""

    def __init__(self, examples_dir: str | None = None):
        """Initialize dataset manager."""
        self.examples_dir = Path(examples_dir) if examples_dir else Path(__file__).parent
        self._dataset_info: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load dataset_info.json."""
        dataset_file = self.examples_dir / "dataset_info.json"
        if dataset_file.exists():
            with open(dataset_file) as f:
                self._dataset_info = json.load(f)

    def get_dataset(self, dataset_name: str) -> dict[str, Any] | None:
        """Get dataset configuration by name."""
        return self._dataset_info.get(dataset_name)

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        return list(self._dataset_info.keys())

    def get_dataset_paths(self, dataset_name: str) -> dict[str, str]:
        """Get dataset paths for command-line arguments."""
        dataset_info = self.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        return {
            "data_path": dataset_info.get("file_name", ""),
            "val_data_path": dataset_info.get("val_file_name", ""),
            "image_root": dataset_info.get("image_root", ""),
        }

    def print_dataset_info(self, dataset_name: str) -> None:
        """Print detailed info for a dataset."""
        dataset_info = self.get_dataset(dataset_name)
        if not dataset_info:
            print(f"Dataset '{dataset_name}' not found")
            return

        print(f"\nDataset: {dataset_name}")
        print(f"  Retrieval mode: {dataset_info.get('retrieval_mode', 'N/A')}")
        print(f"  Data file: {dataset_info.get('file_name', 'N/A')}")
        print(f"  Val file: {dataset_info.get('val_file_name', 'N/A')}")
        if dataset_info.get("image_root"):
            print(f"  Image root: {dataset_info.get('image_root', 'N/A')}")
        if dataset_info.get("columns"):
            print(f"  Columns: {dataset_info['columns']}")


if __name__ == "__main__":
    # Demo usage
    manager = DatasetManager()

    print("Available datasets:")
    for ds in manager.list_datasets():
        info = manager.get_dataset(ds)
        mode = info.get("retrieval_mode", "unknown")
        print(f"  - {ds:<30} ({mode})")

    # Example: Get paths for a dataset
    try:
        print("\nExample: flickr30k_t2i dataset paths:")
        paths = manager.get_dataset_paths("flickr30k_t2i")
        for key, val in paths.items():
            print(f"  {key}: {val}")
        manager.print_dataset_info("flickr30k_t2i")
    except Exception as e:
        print(f"Error: {e}")
