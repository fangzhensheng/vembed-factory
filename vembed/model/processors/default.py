"""Default processor loader — uses AutoProcessor."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoProcessor

from .registry import BaseProcessorLoader, ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("default")
class DefaultProcessorLoader(BaseProcessorLoader):
    """Loads processors via ``AutoProcessor.from_pretrained``.

    This is the catch-all fallback used when no model-specific loader
    matches.
    """

    @staticmethod
    def match(model_name: str) -> bool:  # noqa: ARG004
        # Always matches — acts as fallback.
        return True

    @staticmethod
    def load(model_name: str, **kwargs: Any):
        try:
            return AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("AutoProcessor.from_pretrained failed for '%s': %s", model_name, exc)
            raise
