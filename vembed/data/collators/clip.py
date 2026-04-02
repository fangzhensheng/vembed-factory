"""CLIP-family collator for dual-encoder vision-language models."""

from ..registry import CollatorRegistry
from .base import BaseRetrievalCollator


@CollatorRegistry.register("default")
@CollatorRegistry.register("clip")
@CollatorRegistry.register("siglip")
class CLIPFamilyCollator(BaseRetrievalCollator):
    """Collator for CLIP-family models.

    Registered as "default" collator because CLIP provides a general-purpose
    architecture for paired text-image retrieval that works across diverse domains.
    Aliases: "default", "clip", "siglip"
    """

    def __call__(self, batch):
        fields = self._detect_fields(batch)
        labels = self._extract_labels(batch)

        batch_output = {}

        # Process query text
        if fields["has_query_text"]:
            query_texts = [item["query_text"] for item in batch]
            try:
                query_inputs = self._process_text(query_texts)
            except (TypeError, AttributeError):
                if any((t or "").strip() for t in query_texts):
                    raise
                query_inputs = {"input_ids": None, "attention_mask": None}

            batch_output["input_ids"] = query_inputs.get("input_ids")
            batch_output["attention_mask"] = query_inputs.get("attention_mask")
            batch_output["query_input_ids"] = query_inputs.get("input_ids")
            batch_output["query_attention_mask"] = query_inputs.get("attention_mask")
        else:
            batch_output["input_ids"] = None
            batch_output["attention_mask"] = None
            batch_output["query_input_ids"] = None
            batch_output["query_attention_mask"] = None

        # Process positive text
        if fields["has_pos_text"]:
            pos_texts = [item.get("pos_text", "") for item in batch]
            needs_pos_text = any(
                (t or "").strip() and not self._looks_like_image_path(t) for t in pos_texts
            )
            if needs_pos_text:
                pos_inputs = self._process_text(pos_texts)
                batch_output["pos_input_ids"] = pos_inputs.get("input_ids")
                batch_output["pos_attention_mask"] = pos_inputs.get("attention_mask")
            else:
                batch_output["pos_input_ids"] = None
                batch_output["pos_attention_mask"] = None
        else:
            batch_output["pos_input_ids"] = None
            batch_output["pos_attention_mask"] = None

        # Optimization: Batch process all images in one call
        if fields["has_pos_image"] or fields["has_query_image"]:
            image_groups = []
            group_keys = []

            # Collect all image groups
            if fields["has_pos_image"]:
                pos_images = [item["pos_image"] for item in batch]
                image_groups.append(pos_images)
                group_keys.append("pos_image")

            if fields["has_query_image"]:
                query_images = [item.get("query_image") for item in batch]
                filled = self._fill_query_images(query_images)
                image_groups.append(filled)
                group_keys.append("query_image")

            # Process all groups in one call
            if image_groups:
                results = self._process_images_batch(image_groups)

                # Extract results
                for key, result in zip(group_keys, results, strict=True):
                    pixels = self._get_pixels(result)
                    if key == "pos_image":
                        batch_output["pixel_values"] = pixels
                        batch_output["pos_pixel_values"] = pixels
                    elif key == "query_image":
                        batch_output["query_pixel_values"] = pixels
        else:
            batch_output["pixel_values"] = None
            batch_output["pos_pixel_values"] = None

        if labels is not None:
            batch_output["labels"] = labels

        # Process hard negatives
        neg_images, neg_counts = self._process_negatives(batch, "neg_images")
        if neg_images is not None:
            # Negatives are all images, process as single group
            neg_image_inputs = self._process_images(neg_images)
            batch_output["neg_pixel_values"] = self._get_pixels(neg_image_inputs)
            batch_output["neg_counts"] = neg_counts

        return batch_output
