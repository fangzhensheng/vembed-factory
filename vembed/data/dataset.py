import contextlib
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .loading import load_data

logger = logging.getLogger(__name__)


class GenericRetrievalDataset(Dataset):
    """
    Dataset for visual retrieval tasks supporting various input formats.

    Handles loading of query-positive-negative triplets or pairs from JSONL/CSV/HF Datasets.
    """

    def __init__(
        self,
        data_source: str | list[dict[str, Any]] | Any,
        processor: Any = None,
        image_root: str = "",
        mode: str = "train",
        column_mapping: dict[str, str] | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: File path or loaded data object.
            processor: Processor for tokenization/image processing.
            image_root: Base directory for relative image paths.
            mode: 'train' or 'eval'.
            column_mapping: Optional mapping for dataset columns.
        """
        self.processor = processor
        self.image_root = Path(image_root)
        self.mode = mode
        self.column_mapping = column_mapping or {}

        if isinstance(data_source, str):
            # load_data returns list[dict] or HF Dataset
            self.data = load_data(data_source)
        else:
            self.data = data_source

        logger.info("Initialized dataset with %d examples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, img_input: str | Image.Image) -> Image.Image:
        """Load an image from a path or return existing Image object."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if not img_input:
            # Return a black image as fallback
            return Image.new("RGB", (224, 224), (0, 0, 0))

        # Handle both absolute and relative paths
        img_path = Path(img_input)
        if not img_path.is_absolute() and str(self.image_root):
            full_path = self.image_root / img_path
        else:
            full_path = img_path

        try:
            img = Image.open(full_path).convert("RGB")
            # DEBUG: Print loaded image size - Use print for immediate visibility in benchmark logs
            # if "flickr30k" in str(full_path):
            #    print(f"DEBUG: Loaded image {full_path} size: {img.size}")
            return img
        except (OSError, ValueError) as exc:
            logger.error("Error loading image %s: %s", full_path, exc)
            return Image.new("RGB", (224, 224), (0, 0, 0))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.data[idx]

        # Resolve column names
        query_key = self.column_mapping.get("query", "query")
        query_image_key = self.column_mapping.get("query_image", "query_image")
        positive_key = self.column_mapping.get("positive", "positive")
        negative_key = self.column_mapping.get("negatives", "negatives")

        # Fallback for standard caption/image datasets
        if query_key not in record and "caption" in record:
            query_key = "caption"
        if positive_key not in record and "image" in record:
            positive_key = "image"

        query_text = record.get(query_key, "")
        query_image_path = record.get(query_image_key, "")
        positive_content = record.get(positive_key, "")

        # Load images
        positive_image = self._load_image(positive_content)
        query_image = None

        if query_image_path:
            query_image = self._load_image(query_image_path)

        result = {
            "query_text": query_text,
            "pos_image": positive_image,
            "pos_text": str(positive_content),
        }

        # Try to find a label/class_id for false negative cancellation
        label = record.get("label", record.get("class_id", record.get("id")))
        if label is not None:
            with contextlib.suppress(ValueError, TypeError):
                result["label"] = int(label)

        if query_image is not None:
            result["query_image"] = query_image

        # Load negatives for training
        if self.mode == "train" and negative_key in record:
            negative_inputs = record.get(negative_key, [])
            if isinstance(negative_inputs, str):
                negative_inputs = [negative_inputs]

            negative_images = [self._load_image(p) for p in negative_inputs]
            result["neg_images"] = negative_images

        return result


# Alias for backward compatibility
VisualRetrievalDataset = GenericRetrievalDataset
