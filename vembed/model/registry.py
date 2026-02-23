from .base import BaseEmbeddingModel


class ModelRegistry:
    """
    Registry for VEmbed models.
    Allows users to register custom model architectures.
    """

    _registry: dict[str, type[BaseEmbeddingModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class.
        @ModelRegistry.register("my_custom_model")
        class MyModel(BaseEmbeddingModel): ...
        """

        def decorator(model_cls: type[BaseEmbeddingModel]):
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseEmbeddingModel]:
        if name not in cls._registry:
            raise ValueError(
                f"Model type '{name}' not found in registry. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())
