"""Base collator with shared methods for all retrieval collators."""

import torch
from PIL import Image

from ..utils import looks_like_image_path


class BaseRetrievalCollator:
    """Base class for retrieval collators."""

    def __init__(
        self,
        processor=None,
        mode="train",
        retrieval_mode="t2i",
        text_processor=None,
        image_processor=None,
        **kwargs,
    ):
        self.processor = processor
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.mode = mode
        self.retrieval_mode = retrieval_mode

    def _process_text(self, texts, processor=None):
        """Tokenize texts with fallback."""
        proc = processor or self.processor or self.text_processor
        if proc is None:
            return {"input_ids": None, "attention_mask": None}

        filtered = [t for t in texts if t and str(t).strip()]
        if not filtered:
            return {"input_ids": None, "attention_mask": None}

        # Handle CLIPProcessor: explicitly use `text=...` to avoid ambiguity
        is_processor = hasattr(proc, "image_processor") or hasattr(proc, "process")

        if is_processor:
            return proc(text=filtered, return_tensors="pt", padding=True, truncation=True)
        else:
            return proc(filtered, return_tensors="pt", padding=True, truncation=True)

    def _process_images(self, images, image_processor=None):
        """Process images with processor-aware fallback."""
        proc = image_processor or self.image_processor
        if proc is not None:
            return proc(images=images, return_tensors="pt")

        if self.processor is not None:
            img_proc = getattr(self.processor, "image_processor", None)
            if img_proc is not None:
                return img_proc(images=images, return_tensors="pt")
            return self.processor(images=images, return_tensors="pt")

        raise ValueError("No image processor available")

    @staticmethod
    def _get_pixels(inputs):
        """Extract pixel_values from processed inputs."""
        pixels = inputs.get("pixel_values")
        if pixels is not None:
            return pixels
        return inputs.get("image_tensor")

    @staticmethod
    def _looks_like_image_path(text: str) -> bool:
        """Check if text looks like an image file path."""
        return looks_like_image_path(text)

    def _extract_labels(self, batch):
        """Extract labels if all items have them."""
        labels = [item.get("label") for item in batch]
        has_labels = all(lbl is not None for lbl in labels)
        if has_labels:
            return torch.tensor(labels, dtype=torch.long)
        return None

    def _process_negatives(self, batch, neg_field, processor=None):
        """Extract and process negatives from batch."""
        if self.mode != "train" or neg_field not in batch[0]:
            return None, None

        all_negs = []
        neg_counts = []

        for item in batch:
            negs = item.get(neg_field, [])
            if isinstance(negs, str):
                negs = [negs]
            all_negs.extend(negs)
            neg_counts.append(len(negs))

        if not all_negs:
            return None, None

        if neg_field == "neg_texts" and processor is not None:
            return self._process_text(all_negs, processor), torch.tensor(neg_counts)
        elif neg_field == "neg_images":
            return self._process_images(all_negs), torch.tensor(neg_counts)
        else:
            return all_negs, torch.tensor(neg_counts)

    def _fill_query_images(self, query_images, size=(224, 224), color=(0, 0, 0)):
        """Fill None query images with black placeholders."""
        return [q if q is not None else Image.new("RGB", size, color) for q in query_images]

    @staticmethod
    def _detect_fields(batch):
        """Detect which fields are present in batch."""
        return {
            "has_query_text": any(item.get("query_text") for item in batch),
            "has_pos_text": any(item.get("pos_text") for item in batch),
            "has_pos_image": any(item.get("pos_image") for item in batch),
            "has_query_image": any(item.get("query_image") for item in batch),
        }
