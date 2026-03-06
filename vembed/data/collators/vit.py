"""Vision-only collator for ViT-family models."""

from ..registry import CollatorRegistry
from .base import BaseRetrievalCollator


@CollatorRegistry.register("dino")
@CollatorRegistry.register("dinov2")
@CollatorRegistry.register("dinov3")
@CollatorRegistry.register("vit")
@CollatorRegistry.register("mae")
class VITFamilyCollator(BaseRetrievalCollator):
    """Collator for vision-only models (ViT, DINO, MAE).

    Registry keys: dino, dinov2, dinov3, vit, mae
    """

    def __call__(self, batch):
        fields = self._detect_fields(batch)
        labels = self._extract_labels(batch)

        batch_output = {}

        # Text fields set to None (vision-only)
        batch_output["input_ids"] = None
        batch_output["attention_mask"] = None
        batch_output["query_input_ids"] = None
        batch_output["query_attention_mask"] = None
        batch_output["pos_input_ids"] = None
        batch_output["pos_attention_mask"] = None

        # Process positive image
        if fields["has_pos_image"]:
            pos_images = [item["pos_image"] for item in batch]
            pos_image_inputs = self._process_images(pos_images)
            pos_pixels = self._get_pixels(pos_image_inputs)

            batch_output["pixel_values"] = pos_pixels
            batch_output["pos_pixel_values"] = pos_pixels

        # Process query image
        if fields["has_query_image"]:
            query_images = [item["query_image"] for item in batch]
            query_image_inputs = self._process_images(query_images)
            batch_output["query_pixel_values"] = self._get_pixels(query_image_inputs)

        if labels is not None:
            batch_output["labels"] = labels

        # Process hard negatives
        neg_images, neg_counts = self._process_negatives(batch, "neg_images")
        if neg_images is not None:
            neg_image_inputs = self._process_images(neg_images)
            batch_output["neg_pixel_values"] = self._get_pixels(neg_image_inputs)
            batch_output["neg_counts"] = neg_counts

        return batch_output
