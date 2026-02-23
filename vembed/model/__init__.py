# Register models from backbones directory
# Trigger processor loader registration
from . import backbones, processors
from .base import BaseEmbeddingModel
from .modeling import VisualRetrievalModel
from .processors.registry import ProcessorRegistry
from .registry import ModelRegistry
