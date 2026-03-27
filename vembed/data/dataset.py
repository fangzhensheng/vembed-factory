import contextlib
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .loading import load_data
from .utils import looks_like_image_path

logger = logging.getLogger(__name__)

# Column name aliases for flexible dataset support
COLUMN_ALIASES = {
    "query": ["query", "caption", "text", "question", "instruction", "prompt", "query_text"],
    "positive": ["positive", "image", "answer", "content", "document", "paragraph", "pos_text"],
    "negatives": ["negatives", "negative_samples", "hard_negatives", "distractors"],
    "query_image": ["query_image", "source_image"],
}


class GenericRetrievalDataset(Dataset):
    """
    Dataset for retrieval tasks supporting various input formats.

    Handles loading of query-positive-negative triplets or pairs from JSONL/CSV/HF Datasets.
    Supports text-to-text (t2t), text-to-image (t2i), and image-to-image (i2i) modes.
    Auto-detects whether positive content is text or an image path.
    """

    def __init__(
        self,
        data_source: str | list[dict[str, Any]] | Any,
        processor: Any = None,
        image_root: str = "",
        mode: str = "train",
        column_mapping: dict[str, str] | None = None,
        enable_image_cache: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: File path or loaded data object.
            processor: Processor for tokenization/image processing.
            image_root: Base directory for relative image paths.
            mode: 'train' or 'eval'.
            column_mapping: Optional mapping for dataset columns.
            enable_image_cache: Cache images in memory for faster multi-epoch training.
        """
        self.processor = processor
        self.image_root = Path(image_root)
        self.mode = mode
        self.column_mapping = column_mapping or {}
        self.enable_image_cache = enable_image_cache
        self._image_cache = {}
        self._resolved_keys = {}

        if isinstance(data_source, str):
            self.data = load_data(data_source)
        else:
            self.data = data_source

        logger.info("Initialized dataset with %d examples", len(self.data))
        if self.enable_image_cache:
            logger.info("Image caching enabled for faster multi-epoch training")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, img_input: str) -> Path:
        """Resolve image path relative to image_root."""
        img_path = Path(img_input)
        if not img_path.is_absolute() and str(self.image_root):
            return self.image_root / img_path
        return img_path

    def _load_image(self, img_input: str | Image.Image) -> tuple[Image.Image, bool]:
        """Load an image from a path. Returns (image, success_flag)."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB"), True

        if not img_input:
            # Return a black image as fallback
            return Image.new("RGB", (224, 224), (0, 0, 0)), False

        # Check cache first if enabled
        if self.enable_image_cache and str(img_input) in self._image_cache:
            return self._image_cache[str(img_input)].copy(), True

        full_path = self._resolve_path(img_input)

        try:
            img = Image.open(full_path).convert("RGB")
            # Cache if enabled
            if self.enable_image_cache:
                self._image_cache[str(img_input)] = img
            return img, True
        except (OSError, ValueError) as exc:
            logger.error("Error loading image %s: %s", full_path, exc)
            return Image.new("RGB", (224, 224), (0, 0, 0)), False

    @staticmethod
    def _looks_like_image_path(content: str) -> bool:
        """Check if content looks like an image file path."""
        return looks_like_image_path(content)

    def _resolve_column_name(self, key: str, record: dict[str, Any]) -> str:
        """Resolve column name using mapping, aliases, or default value.

        Priority:
        1. Explicit column_mapping
        2. Auto-detect via COLUMN_ALIASES
        3. Default fallback
        """
        if key in self._resolved_keys:
            return self._resolved_keys[key]

        mapped_key = self.column_mapping.get(key)
        if mapped_key:
            self._resolved_keys[key] = mapped_key
            return mapped_key

        aliases = COLUMN_ALIASES.get(key, [])
        for alias in aliases:
            if alias in record:
                self._resolved_keys[key] = alias
                return alias

        default_key = {"query": "query", "positive": "positive", "negatives": "negatives"}.get(
            key, key
        )
        self._resolved_keys[key] = default_key

        return default_key

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.data[idx]

        query_key = self._resolve_column_name("query", record)
        positive_key = self._resolve_column_name("positive", record)
        negative_key = self._resolve_column_name("negatives", record)
        query_image_key = self.column_mapping.get("query_image", "query_image")

        query_text = record.get(query_key, "")
        query_image_path = record.get(query_image_key, "")
        positive_content = record.get(positive_key, "")

        # Detect if positive content is text (t2t) or image path (t2i/i2i)
        is_text_positive = self._looks_like_image_path(positive_content) is False

        result = {
            "query_text": query_text,
            "pos_text": str(positive_content),
        }

        # Load images only for visual retrieval modes
        query_image = None
        if query_image_path:
            query_image, q_success = self._load_image(query_image_path)
            # Only set path if load successful to avoid downstream errors
            if q_success:
                result["query_image_path"] = str(self._resolve_path(query_image_path))

        if not is_text_positive:
            # Visual retrieval: load positive as image
            pos_img, p_success = self._load_image(positive_content)
            result["pos_image"] = pos_img
            if p_success:
                result["pos_image_path"] = str(self._resolve_path(positive_content))
        else:
            # Text retrieval: skip image loading (avoids unnecessary black placeholder)
            result["pos_image"] = None

        if query_image is not None:
            result["query_image"] = query_image

        # Try to find a label/class_id for false negative cancellation
        label = record.get("label", record.get("class_id", record.get("id")))
        if label is not None:
            with contextlib.suppress(ValueError, TypeError):
                result["label"] = int(label)

        # Load negatives for training
        if self.mode == "train" and negative_key in record:
            negative_inputs = record.get(negative_key, [])
            if isinstance(negative_inputs, str):
                negative_inputs = [negative_inputs]

            if is_text_positive:
                # Text-to-text: negatives are text passages
                result["neg_texts"] = negative_inputs
            else:
                # Visual retrieval: negatives are images
                neg_results = [self._load_image(p) for p in negative_inputs]
                negative_images = [res[0] for res in neg_results]
                result["neg_images"] = negative_images

                # Only include paths for successfully loaded images
                neg_paths = []
                for p, (_, success) in zip(negative_inputs, neg_results, strict=True):
                    if success:
                        neg_paths.append(str(self._resolve_path(p)))
                    else:
                        neg_paths.append(None)
                result["neg_image_paths"] = neg_paths

        return result


# Alias for backward compatibility
VisualRetrievalDataset = GenericRetrievalDataset
