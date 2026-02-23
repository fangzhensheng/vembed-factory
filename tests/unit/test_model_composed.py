from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from vembed.model.modeling import VisualRetrievalModel


def test_model_composed_forward():
    model = VisualRetrievalModel(
        model_name_or_path="bert-base-uncased",  # unused in composed
        encoder_mode="composed",
        text_model_name="bert-base-uncased",
        image_model_name="facebook/dinov2-base",
    )
    # Text path
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    t_inputs = tok(["hello world"], return_tensors="pt")
    t_out = model(input_ids=t_inputs["input_ids"], attention_mask=t_inputs["attention_mask"])
    assert t_out.shape[0] == 1
    # Image path
    img_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    img = Image.new("RGB", (224, 224), (0, 0, 255))
    i_inputs = img_proc(images=[img], return_tensors="pt")
    i_out = model(pixel_values=i_inputs["pixel_values"])
    assert i_out.shape[0] == 1
