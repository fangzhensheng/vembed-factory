import torch.nn as nn


class LossRegistry:
    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(loss_cls: type[nn.Module]):
            cls._registry[name] = loss_cls
            return loss_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[nn.Module]:
        if name not in cls._registry:
            raise ValueError(
                f"Loss '{name}' is not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_losses(cls):
        return list(cls._registry.keys())
