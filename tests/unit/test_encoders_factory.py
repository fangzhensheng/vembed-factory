import torch
from PIL import Image

from vembed.model.encoders_factory import SimpleImageEncoder, SimpleTextEncoder


def test_text_encoder_basic():
    enc = SimpleTextEncoder("bert-base-uncased", pooling="mean")
    out = enc.encode(["hello world"])
    assert out.shape[0] == 1
    assert torch.isfinite(out).all()


def test_image_encoder_basic():
    enc = SimpleImageEncoder("facebook/dinov2-base", pooling="cls")
    img = Image.new("RGB", (224, 224), (0, 255, 0))
    out = enc.encode([img])
    assert out.shape[0] == 1
    assert torch.isfinite(out).all()
