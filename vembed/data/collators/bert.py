"""BERT-family collator for text-only encoder models."""

from ..registry import CollatorRegistry
from .base import BaseRetrievalCollator


@CollatorRegistry.register("bge")
@CollatorRegistry.register("bert")
@CollatorRegistry.register("e5")
@CollatorRegistry.register("qwen")
class BERTFamilyCollator(BaseRetrievalCollator):
    """Collator for BERT-family text encoder models."""

    def __call__(self, batch):
        fields = self._detect_fields(batch)
        labels = self._extract_labels(batch)

        batch_output = {}

        # Image fields set to None (text-only)
        batch_output["pixel_values"] = None
        batch_output["pos_pixel_values"] = None
        batch_output["query_pixel_values"] = None

        # Process query text
        if fields["has_query_text"]:
            query_texts = [item["query_text"] for item in batch]
            query_inputs = self._process_text(query_texts)

            batch_output["input_ids"] = query_inputs.get("input_ids")
            batch_output["attention_mask"] = query_inputs.get("attention_mask")
            batch_output["query_input_ids"] = query_inputs.get("input_ids")
            batch_output["query_attention_mask"] = query_inputs.get("attention_mask")

        # Process positive text
        if fields["has_pos_text"]:
            pos_texts = [item.get("pos_text", "") for item in batch]
            pos_inputs = self._process_text(pos_texts)

            batch_output["pos_input_ids"] = pos_inputs.get("input_ids")
            batch_output["pos_attention_mask"] = pos_inputs.get("attention_mask")

        if labels is not None:
            batch_output["labels"] = labels

        # Process hard negatives
        neg_texts, neg_counts = self._process_negatives(
            batch, "neg_texts", processor=self.processor
        )
        if neg_texts is not None:
            batch_output["neg_input_ids"] = neg_texts.get("input_ids")
            batch_output["neg_attention_mask"] = neg_texts.get("attention_mask")
            batch_output["neg_counts"] = neg_counts

        return batch_output
