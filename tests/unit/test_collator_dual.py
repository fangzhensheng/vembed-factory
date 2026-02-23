# from transformers import AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image

from vembed.data.collators.default import VisualRetrievalCollator


class MockProcessor:
    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, truncation=True):
        if text is not None:
            # Return dict with input_ids and attention_mask
            return {
                "input_ids": torch.ones(len(text), 10),
                "attention_mask": torch.ones(len(text), 10),
            }
        if images is not None:
            # Return dict with pixel_values
            return {"pixel_values": torch.ones(len(images), 3, 224, 224)}
        return {}


def test_collator_dual_processors():
    text_proc = MockProcessor()
    image_proc = MockProcessor()
    coll = VisualRetrievalCollator(
        processor=None,
        mode="train",
        text_processor=text_proc,
        image_processor=image_proc,
    )

    # Test batch with pos_text (for i2t/t2t support)
    batch = [
        {
            "query_text": "a green square",
            "pos_image": Image.new("RGB", (224, 224), (0, 255, 0)),
            "pos_text": "green color",
        },
        {
            "query_text": "a red square",
            "pos_image": Image.new("RGB", (224, 224), (255, 0, 0)),
            "pos_text": "red color",
        },
    ]
    out = coll(batch)

    assert "input_ids" in out
    assert "pixel_values" in out

    # Verify new fields for t2t
    assert "pos_input_ids" in out
    assert "pos_attention_mask" in out

    # Check shapes
    assert out["input_ids"].shape[0] == 2
    assert out["pos_input_ids"].shape[0] == 2
