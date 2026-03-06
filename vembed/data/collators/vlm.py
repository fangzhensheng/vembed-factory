"""Universal VLM collator for chat-template based vision-language models."""

import abc
import logging
import unicodedata
from typing import Any

import torch

from ..registry import CollatorRegistry
from .base import BaseRetrievalCollator

logger = logging.getLogger(__name__)

try:
    from qwen_vl_utils.vision_process import process_vision_info

    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False


class VLMStrategy(abc.ABC):
    """Abstract strategy for handling different VLM families."""

    def __init__(self, processor, prompt=""):
        self.processor = processor
        self.prompt = prompt

    @abc.abstractmethod
    def format_conversation(
        self,
        text: str | None,
        image_input: Any | None,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Format a single sample into a conversation list."""
        pass

    @abc.abstractmethod
    def process_batch(
        self, conversations: list[list[dict[str, Any]]], images: list[Any]
    ) -> dict[str, Any]:
        """Process a batch of conversations and images into model inputs."""
        pass


class QwenVLMStrategy(VLMStrategy):
    """Strategy for Qwen-VL family models."""

    def __init__(self, processor, prompt="Represent the user's input."):
        super().__init__(processor, prompt)
        if not QWEN_UTILS_AVAILABLE:
            logger.warning("qwen_vl_utils not installed. Qwen strategy might fail.")

    def format_conversation(
        self,
        text: str | None,
        image_input: Any | None,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        instr = instruction or self.prompt
        if instr:
            instr = instr.strip()
            if instr and not unicodedata.category(instr[-1]).startswith("P"):
                instr = instr + "."

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instr}]},
            {"role": "user", "content": content},
        ]

        if image_input is not None:
            if isinstance(image_input, str) and not image_input.startswith(
                ("http", "oss", "file://")
            ):
                image_input = "file://" + image_input

            content.append(
                {
                    "type": "image",
                    "image": image_input,
                }
            )

        if text:
            content.append({"type": "text", "text": text})
        elif not image_input:
            content.append({"type": "text", "text": "NULL"})

        return conversation

    def process_batch(
        self, conversations: list[list[dict[str, Any]]], images: list[Any]
    ) -> dict[str, Any]:
        text_inputs = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )

        try:
            image_inputs, video_inputs = process_vision_info(conversations)
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            image_inputs, video_inputs = None, None

        return self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


class GenericVLMStrategy(VLMStrategy):
    """Strategy for generic VLMs (LLaVA, etc.) using standard HF APIs."""

    def format_conversation(
        self,
        text: str | None,
        image_input: Any | None,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        content = []
        conversation = [{"role": "user", "content": content}]

        if image_input is not None:
            content.append({"type": "image"})

        final_text = text
        if not final_text and image_input is not None:
            final_text = self.prompt

        if final_text:
            content.append({"type": "text", "text": final_text})
        else:
            content.append({"type": "text", "text": " "})

        return conversation

    def process_batch(
        self, conversations: list[list[dict[str, Any]]], images: list[Any]
    ) -> dict[str, Any]:
        text_inputs = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )

        # Filter out None images to ensure compatibility with standard HF processors
        # which expect a list of valid image objects corresponding to <image> tokens
        valid_images = [img for img in images if img is not None]

        return self.processor(
            text=text_inputs,
            images=valid_images if valid_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


@CollatorRegistry.register("qwen-vl")
@CollatorRegistry.register("llava")
@CollatorRegistry.register("phi-3-vision")
class VLMRetrievalCollator(BaseRetrievalCollator):
    """Collator for Vision-Language Models with strategy pattern support.

    Delegates VLM-specific logic to strategy classes based on processor type.
    """

    def __init__(self, processor, mode="train", prompt="Represent the user's input.", **kwargs):
        super().__init__(processor, mode, **kwargs)
        self.prompt = prompt
        self.strategy = self._get_strategy(processor, prompt)

        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer:
            self.processor.tokenizer.padding_side = "right"

    def _get_strategy(self, processor, prompt) -> VLMStrategy:
        """Select strategy based on processor type."""
        if processor is None:
            return GenericVLMStrategy(processor, prompt)

        class_name = processor.__class__.__name__.lower()
        model_name = getattr(processor, "name", "").lower()

        if "qwen" in class_name or "qwen" in model_name:
            return QwenVLMStrategy(processor, prompt)

        return GenericVLMStrategy(processor, prompt)

    def _process_batch_chunk(
        self,
        texts: list[str | None],
        images: list[Any],
        image_paths: list[str | None],
    ) -> dict[str, Any]:
        """Process a batch using the selected strategy."""
        if not any(texts) and not any(images):
            return {}

        conversations = []
        # Keep images aligned with conversations
        aligned_images = [None] * len(texts)

        for i, text in enumerate(texts):
            img = images[i] if images and i < len(images) else None
            img_path = image_paths[i] if image_paths and i < len(image_paths) else None

            img_input = img_path if isinstance(self.strategy, QwenVLMStrategy) and img_path else img

            if img is not None:
                aligned_images[i] = img

            conv = self.strategy.format_conversation(text, img_input)
            conversations.append(conv)

        return self.strategy.process_batch(conversations, aligned_images)

    def __call__(self, batch):
        batch_output = {}

        q_texts = [item.get("query_text") for item in batch]
        q_images = [item.get("query_image") for item in batch]
        q_paths = [item.get("query_image_path") for item in batch]

        if any(t or i for t, i in zip(q_texts, q_images, strict=True)):
            q_out = self._process_batch_chunk(q_texts, q_images, q_paths)

            batch_output["query_input_ids"] = q_out.get("input_ids")
            batch_output["query_attention_mask"] = q_out.get("attention_mask")
            if "pixel_values" in q_out:
                batch_output["query_pixel_values"] = q_out["pixel_values"]
            if "image_grid_thw" in q_out:
                batch_output["query_image_grid_thw"] = q_out["image_grid_thw"]

            batch_output["input_ids"] = batch_output["query_input_ids"]
            batch_output["attention_mask"] = batch_output["query_attention_mask"]

        p_texts_raw = [item.get("pos_text") for item in batch]
        p_images = [item.get("pos_image") for item in batch]
        p_paths = [item.get("pos_image_path") for item in batch]

        p_texts = []
        for t, img in zip(p_texts_raw, p_images, strict=True):
            # If there's an image, use prompt as fallback; otherwise use pos_text
            p_texts.append(self.prompt if img is not None else t)

        if any(t or i for t, i in zip(p_texts, p_images, strict=True)):
            p_out = self._process_batch_chunk(p_texts, p_images, p_paths)

            batch_output["pos_input_ids"] = p_out.get("input_ids")
            batch_output["pos_attention_mask"] = p_out.get("attention_mask")
            if "pixel_values" in p_out:
                batch_output["pos_pixel_values"] = p_out["pixel_values"]
                batch_output["pixel_values"] = p_out["pixel_values"]
            if "image_grid_thw" in p_out:
                batch_output["pos_image_grid_thw"] = p_out["image_grid_thw"]

        if self.mode == "train":
            neg_images = []
            neg_paths = []
            neg_counts = []

            for item in batch:
                n_imgs = item.get("neg_images", [])
                n_paths = item.get("neg_image_paths", [])
                # Always track counts to maintain batch size alignment
                neg_counts.append(len(n_imgs))
                if n_imgs:
                    neg_images.extend(n_imgs)
                    neg_paths.extend(n_paths if n_paths else [None] * len(n_imgs))

            if neg_images:
                n_texts = [None] * len(neg_images)
                n_out = self._process_batch_chunk(n_texts, neg_images, neg_paths)

                batch_output["neg_pixel_values"] = n_out.get("pixel_values")
                if "image_grid_thw" in n_out:
                    batch_output["neg_image_grid_thw"] = n_out["image_grid_thw"]
                if "input_ids" in n_out:
                    batch_output["neg_input_ids"] = n_out["input_ids"]
                    batch_output["neg_attention_mask"] = n_out["attention_mask"]

            batch_output["neg_counts"] = torch.tensor(neg_counts)

        return batch_output
