"""Processor loading with a pluggable registry.

Register new model-specific loaders by adding a file to this package and
decorating the class with ``@ProcessorRegistry.register("my_mode")``.

Backward-compatible convenience functions
(:func:`build_multimodal_processor`, :func:`build_text_processor`,
:func:`build_image_processor`) are re-exported at package level.
"""

from __future__ import annotations

import importlib
import os
import pkgutil

from transformers import AutoImageProcessor, AutoTokenizer

from .registry import BaseProcessorLoader, ProcessorRegistry  # noqa: F401

_package_dir = os.path.dirname(__file__)
for _finder, _module_name, _ispkg in pkgutil.iter_modules([_package_dir]):
    if _module_name not in ("__init__", "registry"):
        importlib.import_module(f".{_module_name}", package=__name__)


def build_multimodal_processor(model_name: str, **kwargs):
    """Build the multimodal processor for *model_name*.

    Delegates to :meth:`ProcessorRegistry.resolve` which auto-detects
    the right loader based on the model name.
    """
    return ProcessorRegistry.resolve(model_name, **kwargs)


def build_text_processor(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)


def build_image_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
