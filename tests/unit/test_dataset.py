import json
import os

from PIL import Image
from transformers import AutoProcessor

from vembed.data.collators.default import VisualRetrievalCollator
from vembed.data.dataset import VisualRetrievalDataset


def test_jsonl_dataset_tmp(tmp_path):
    # Create minimal dataset
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_path = images_dir / "a.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(img_path)

    jsonl_path = tmp_path / "train.jsonl"
    with open(jsonl_path, "w") as f:
        entry = {"query": "a red square", "positive": os.path.join("images", "a.jpg")}
        f.write(json.dumps(entry) + "\n")

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ds = VisualRetrievalDataset(
        str(jsonl_path), processor=processor, image_root=str(tmp_path), mode="train"
    )
    coll = VisualRetrievalCollator(processor, mode="train")

    sample = ds[0]
    assert "query_text" in sample and "pos_image" in sample

    batch = [ds[0]] * 2
    batch_out = coll(batch)
    assert "input_ids" in batch_out and "pixel_values" in batch_out
