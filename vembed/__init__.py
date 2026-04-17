"""vembed-factory: A Factory for Visual & Multimodal Embeddings.

Quick Start::

    from vembed import Trainer, VEmbedModel

    # Train using Python API
    trainer = Trainer("openai/clip-vit-base-patch32", output_dir="output")
    trainer.train(data_path="data/train.jsonl", epochs=3)

    # Use for inference
    predictor = VEmbedModel(model_path="output/checkpoint-epoch-3")
    emb = predictor.encode_text("a photo of a cat")
"""

__version__ = "0.1.0"
__author__ = "Fang Zhensheng"

__all__ = [
    "Trainer",
    "VEmbedTrainer",
    "VEmbedModel",
    "VisualRetrievalModel",
    "LossFactory",
    "__version__",
]


def __getattr__(name: str):
    if name == "VEmbedModel":
        from vembed.inference import VEmbedModel

        return VEmbedModel
    if name in ("Trainer", "VEmbedTrainer"):
        from vembed.trainer import VEmbedTrainer

        return VEmbedTrainer
    if name == "VisualRetrievalModel":
        from vembed.model.modeling import VisualRetrievalModel

        return VisualRetrievalModel
    if name == "LossFactory":
        from vembed.losses.factory import LossFactory

        return LossFactory
    raise AttributeError(f"module 'vembed' has no attribute {name!r}")
