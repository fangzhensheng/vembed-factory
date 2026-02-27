import torch.nn as nn


class LossRegistry:
    """Registry for loss function modules.

    Manages registration and retrieval of loss functions used in training.
    """

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a loss function class.

        Args:
            name: Unique identifier for the loss function.

        Returns:
            Decorator function that registers the loss class.
        """
        def decorator(loss_cls: type[nn.Module]):
            cls._registry[name] = loss_cls
            return loss_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[nn.Module]:
        """Retrieve a registered loss function class.

        Args:
            name: Name of the loss function to retrieve.

        Returns:
            The loss class.

        Raises:
            ValueError: If the loss name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Loss '{name}' is not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_losses(cls):
        """List all registered loss function names.

        Returns:
            List of registered loss function names.
        """
        return list(cls._registry.keys())
