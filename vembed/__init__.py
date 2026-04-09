"""vembed-factory: A Factory for Visual & Multimodal Embeddings.

Quick Start::

    from vembed import Trainer, VEmbedModel

    # Train using Python API
    trainer = Trainer("openai/clip-vit-base-patch32")
    trainer.train(data_path="data/train.jsonl", output_dir="output", epochs=3)

    # Use for inference
    predictor = VEmbedModel(model_path="output/checkpoint-epoch-3")
    emb = predictor.encode_text("a photo of a cat")
"""

__version__ = "0.1.0"
__author__ = "Fang Zhensheng"

from vembed.inference import VEmbedModel  # noqa: F401
from vembed.losses.factory import LossFactory  # noqa: F401
from vembed.model.modeling import VisualRetrievalModel  # noqa: F401
from vembed.trainer import VEmbedTrainer  # noqa: F401
from vembed.trainer import VEmbedTrainer as Trainer

__all__ = [
    "Trainer",
    "VEmbedTrainer",
    "VEmbedModel",
    "VisualRetrievalModel",
    "LossFactory",
    "__version__",
]
