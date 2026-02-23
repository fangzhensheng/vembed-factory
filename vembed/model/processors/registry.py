"""Processor registry — maps encoder_mode names to processor loaders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseProcessorLoader(ABC):
    """Base class for model-specific processor loaders.

    Sub-classes must implement:

    * :meth:`match` — return *True* when a model name / path should be handled
      by this loader (used for auto-detection when no explicit *encoder_mode* is
      given).
    * :meth:`load` — actually load and return the processor object.
    """

    @staticmethod
    @abstractmethod
    def match(model_name: str) -> bool:
        """Return *True* if this loader can handle *model_name*."""
        ...

    @staticmethod
    @abstractmethod
    def load(model_name: str, **kwargs: Any):
        """Load and return the processor for *model_name*."""
        ...


class ProcessorRegistry:
    """Registry for processor loaders, keyed by *encoder_mode* name.

    Usage::

        @ProcessorRegistry.register("qwen3_vl")
        class Qwen3VLProcessorLoader(BaseProcessorLoader):
            ...

        # Explicit look-up by encoder_mode
        loader_cls = ProcessorRegistry.get("qwen3_vl")
        processor = loader_cls.load(model_name)

        # Auto-detect (iterates registered loaders in priority order)
        processor = ProcessorRegistry.resolve(model_name)
    """

    _registry: dict[str, type[BaseProcessorLoader]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor loader under *name*."""

        def decorator(loader_cls: type[BaseProcessorLoader]):
            cls._registry[name] = loader_cls
            return loader_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseProcessorLoader] | None:
        """Return the loader class for *name*, or ``None``."""
        return cls._registry.get(name)

    @classmethod
    def list_loaders(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def resolve(cls, model_name: str, encoder_mode: str | None = None, **kwargs: Any):
        """Auto-detect the right loader by calling ``match()`` on each.

        Loaders are tried in registration order; the *default* loader is
        always tried last.  Returns the loaded processor, or raises
        :class:`ValueError` if nothing works.
        """
        # If encoder_mode is explicit, try to find that specific loader first
        if encoder_mode and encoder_mode != "auto" and encoder_mode in cls._registry:
            loader_cls = cls._registry[encoder_mode]
            return loader_cls.load(model_name, **kwargs)

        # Try non-default loaders first
        for name, loader_cls in cls._registry.items():
            if name == "default":
                continue
            if loader_cls.match(model_name):
                try:
                    return loader_cls.load(model_name, **kwargs)
                except Exception as exc:
                    logger.warning(
                        "ProcessorLoader '%s' matched but failed for '%s': %s",
                        name,
                        model_name,
                        exc,
                    )

        # Fallback to default
        default_cls = cls._registry.get("default")
        if default_cls is not None:
            return default_cls.load(model_name, **kwargs)

        raise ValueError(
            f"No processor loader matched for '{model_name}'. " f"Available: {cls.list_loaders()}"
        )
