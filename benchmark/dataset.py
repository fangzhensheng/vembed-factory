"""
Karpathy-split dataset loading for Flickr30k / COCO evaluation.

Handles image deduplication: Flickr30k test has 1000 images with 5 captions each,
so we encode 1000 unique images and 5000 captions, then map each caption to its
ground-truth image index.
"""

import json
import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class KarpathyEntry:
    query: str
    positive: str
    image_id: str = ""


class KarpathyDataset:
    """Loads a jsonl dataset and deduplicates images for retrieval evaluation."""

    def __init__(self, jsonl_path: str, image_root: str):
        self.entries: list[KarpathyEntry] = []
        with open(jsonl_path) as f:
            for line in f:
                raw = json.loads(line)
                self.entries.append(
                    KarpathyEntry(
                        **{k: raw[k] for k in ("query", "positive", "image_id") if k in raw}
                    )
                )

        self.image_root = image_root

        # Deduplicate images and build ground-truth mapping
        self.unique_image_paths: list[str] = []
        self.caption_to_image_idx: list[int] = []
        self._build_index()

    def _build_index(self):
        seen: dict[str, int] = {}
        for entry in self.entries:
            abs_path = os.path.join(self.image_root, os.path.basename(entry.positive))
            if abs_path not in seen:
                seen[abs_path] = len(self.unique_image_paths)
                self.unique_image_paths.append(abs_path)
            self.caption_to_image_idx.append(seen[abs_path])

    @property
    def num_images(self) -> int:
        return len(self.unique_image_paths)

    @property
    def num_captions(self) -> int:
        return len(self.entries)

    def captions(self) -> list[str]:
        return [e.query for e in self.entries]

    def __len__(self):
        return len(self.entries)


class ImageListDataset(Dataset):
    """Wraps a list of image paths for DataLoader batching."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> Image.Image:
        try:
            return Image.open(self.paths[idx]).convert("RGB")
        except (FileNotFoundError, OSError):
            return Image.new("RGB", (224, 224), (0, 0, 0))
