"""SigLIP processor loader.

Forces specific processor configuration for SigLIP models to ensure
correct padding/truncation behavior (max_length=64).
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoProcessor

from .registry import BaseProcessorLoader, ProcessorRegistry

logger = logging.getLogger(__name__)


class SigLIPProcessorWrapper:
    """Wrapper for SigLIP processors to enforce padding configuration.

    This wrapper preserves the original object structure, avoiding stability issues
    associated with class swizzling and multiprocessing (pickle).
    """

    def __init__(self, processor: Any):
        self.processor = processor

    def __getattr__(self, name: str) -> Any:
        # Prevent recursion if 'processor' is missing
        if name == "processor":
            raise AttributeError
        return getattr(self.processor, name)

    def __getstate__(self) -> dict[str, Any]:
        # Explicitly return state for pickling to avoid __getattr__ recursion
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"<SigLIPProcessorWrapper wrapping {repr(self.processor)}>"

    def _is_text_call(self, args: tuple, kwargs: dict) -> bool:
        """Determine if the call involves text processing."""
        if "text" in kwargs:
            return True

        if args:
            first_arg = args[0]
            if isinstance(first_arg, str):
                return True
            if isinstance(first_arg, list) and first_arg and isinstance(first_arg[0], str):
                return True

        return False

    def _handle_image_call(self, images: Any, kwargs: dict) -> Any:
        """Handle image-only processing calls."""
        # Robustly retrieve image_processor from the wrapped processor
        image_processor = getattr(self.processor, "image_processor", None)

        # Fallback checks if standard access fails
        if image_processor is None and hasattr(self.processor, "attributes"):
            attr_name = self.processor.attributes.get("image_processor")
            if attr_name:
                image_processor = getattr(self.processor, attr_name, None)

        if image_processor:
            return image_processor(images, **kwargs)

        # Fallback to calling the processor directly
        if "text" not in kwargs:
            kwargs["text"] = None
        if "text_target" not in kwargs:
            kwargs["text_target"] = None
        return self.processor(images=images, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        is_text_call = self._is_text_call(args, kwargs)

        if is_text_call:
            kwargs["padding"] = "max_length"
            kwargs["truncation"] = True
            kwargs["max_length"] = 64

        # If strictly image processing, bypass ProcessorMixin validation logic
        if not is_text_call and "images" in kwargs:
            images = kwargs.pop("images")
            return self._handle_image_call(images, kwargs)

        return self.processor(*args, **kwargs)


@ProcessorRegistry.register("siglip")
class SigLIPProcessorLoader(BaseProcessorLoader):
    """Loads the processor for SigLIP models with custom configuration wrapper.

    SigLIP models are sensitive to padding. This loader wraps the standard
    AutoProcessor to enforce `padding='max_length'` and `max_length=64`
    when processing text, ensuring consistent embedding quality.
    """

    @staticmethod
    def match(model_name: str) -> bool:
        return "siglip" in model_name.lower()

    @staticmethod
    def load(model_name: str, **kwargs: Any) -> Any:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **kwargs)

        return SigLIPProcessorWrapper(processor)
