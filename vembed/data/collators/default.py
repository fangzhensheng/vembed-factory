import torch
from PIL import Image

from ..registry import CollatorRegistry


@CollatorRegistry.register("default")
class VisualRetrievalCollator:
    def __init__(
        self, processor=None, mode="train", text_processor=None, image_processor=None, **kwargs
    ):
        self.processor = processor
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.mode = mode

    def _process_text(self, texts):
        if self.processor is not None:
            return self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        if self.text_processor is None:
            return {"input_ids": None, "attention_mask": None}
        return self.text_processor(texts, return_tensors="pt", padding=True, truncation=True)

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        lower = (text or "").lower()
        if not lower:
            return False
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")):
            return True
        return "/" in lower or "\\" in lower

    def _process_images(self, images):
        if self.image_processor is not None:
            return self.image_processor(images=images, return_tensors="pt")
        if self.processor is not None:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None:
                return image_processor(images=images, return_tensors="pt")
            return self.processor(images=images, return_tensors="pt")
        return self.image_processor(images=images, return_tensors="pt")

    @staticmethod
    def _get_pixels(inputs):
        pixels = inputs.get("pixel_values")
        if pixels is not None:
            return pixels
        return inputs.get("image_tensor")

    def __call__(self, batch):
        query_texts = [item["query_text"] for item in batch]
        pos_images = [item["pos_image"] for item in batch]
        pos_texts = [item.get("pos_text", "") for item in batch]
        query_images = [item.get("query_image") for item in batch]
        labels = [item.get("label") for item in batch]
        has_labels = all(lbl is not None for lbl in labels)

        try:
            text_inputs = self._process_text(query_texts)
        except TypeError:
            if any((t or "").strip() for t in query_texts):
                raise
            text_inputs = {"input_ids": None, "attention_mask": None}

        image_inputs = self._process_images(pos_images)
        needs_pos_text = any((t or "").strip() and not self._looks_like_path(t) for t in pos_texts)
        if needs_pos_text:
            pos_text_inputs = self._process_text(pos_texts)
        else:
            pos_text_inputs = {"input_ids": None, "attention_mask": None}

        pixels = self._get_pixels(image_inputs)

        batch_output = {
            "input_ids": text_inputs.get("input_ids"),
            "attention_mask": text_inputs.get("attention_mask"),
            "pixel_values": pixels,
            "query_input_ids": text_inputs.get("input_ids"),
            "query_attention_mask": text_inputs.get("attention_mask"),
            "pos_pixel_values": pixels,
            "pos_input_ids": pos_text_inputs.get("input_ids"),
            "pos_attention_mask": pos_text_inputs.get("attention_mask"),
        }

        if any(q is not None for q in query_images):
            # None images become black placeholders so the batch can be stacked
            filled = [
                q if q is not None else Image.new("RGB", (224, 224), (0, 0, 0))
                for q in query_images
            ]
            batch_output["query_pixel_values"] = self._get_pixels(self._process_images(filled))

        if has_labels:
            batch_output["labels"] = torch.tensor(labels, dtype=torch.long)

        if self.mode == "train" and "neg_images" in batch[0]:
            all_neg_images = []
            neg_counts = []
            for item in batch:
                negs = item.get("neg_images", [])
                all_neg_images.extend(negs)
                neg_counts.append(len(negs))

            if all_neg_images:
                batch_output["neg_pixel_values"] = self._get_pixels(
                    self._process_images(all_neg_images)
                )
                batch_output["neg_counts"] = torch.tensor(neg_counts)

        return batch_output
