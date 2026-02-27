import logging
import traceback
import unicodedata

import torch

from ..registry import CollatorRegistry

logger = logging.getLogger(__name__)

# Try to import Qwen utils
try:
    from qwen_vl_utils.vision_process import process_vision_info

    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False


@CollatorRegistry.register("qwen3_vl")
class QwenVisualRetrievalCollator:
    """
    Specialized collator for Qwen-VL that handles conversation format and image grids.
    Refactored to align with Qwen3-VL official logic.
    """

    # Qwen3-VL Official Constants
    IMAGE_BASE_FACTOR = 16
    IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
    # Match Qwen3VLEmbedder reference defaults
    # min_pixels = 4 * 32 * 32 = 4,096
    # max_pixels = 1800 * 32 * 32 = 1,843,200
    MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
    MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR

    def __init__(self, processor, mode="train", prompt="", **kwargs):
        self.processor = processor
        self.mode = mode
        self.prompt = prompt
        # Ensure padding side is right
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer:
            self.processor.tokenizer.padding_side = "right"
        if not QWEN_UTILS_AVAILABLE:
            logger.warning(
                "Qwen utils not available. QwenVisualRetrievalCollator might fail if process_vision_info is needed."
            )

    def _format_model_input(self, text=None, image=None, instruction=None):
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction = instruction + "."

        content = []
        # Note: We rely on the chat template to insert the default system prompt if needed.
        # Reference implementation uses "Represent the user's input." as default.

        conversation = [
            {"role": "user", "content": content},
        ]

        if image:
            # image is PIL.Image or path
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.MIN_PIXELS,
                    "max_pixels": self.MAX_PIXELS,
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        if not image and not text:
            content.append({"type": "text", "text": "NULL"})

        return conversation

    def _process_item(self, text=None, image=None):
        try:
            conversation = self._format_model_input(text=text, image=image, instruction=self.prompt)

            text_input = self.processor.apply_chat_template(
                [conversation], add_generation_prompt=True, tokenize=False
            )

            # Only call process_vision_info when there are visual inputs
            image_inputs = None
            video_kwargs = {}
            if image is not None:
                if not QWEN_UTILS_AVAILABLE:
                    raise RuntimeError(
                        "qwen_vl_utils is required for image processing but not installed. "
                        "Install it with: pip install qwen-vl-utils"
                    )
                image_inputs, _video_inputs, video_kwargs = process_vision_info(
                    [conversation],
                    image_patch_size=16,
                    return_video_metadata=True,
                    return_video_kwargs=True,
                )

            inputs = self.processor(
                text=text_input,
                images=image_inputs if image_inputs else None,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )

            # Squeeze batch dimension added by processor
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            if input_ids.dim() == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)

            pixel_values = inputs.get("pixel_values", None)

            # Handle grid
            grid_thw = None
            if "image_grid_thw" in inputs:
                grid = inputs["image_grid_thw"]
                if grid is not None:
                    # Ensure grid is 2D [N, 3]
                    # If it came from single item processing, it might be [1, 1, 3] or [1, 3]
                    if grid.dim() == 3 and grid.shape[0] == 1:
                        grid_thw = grid.squeeze(0)
                    elif grid.dim() == 2:
                        grid_thw = grid
                    elif grid.dim() == 1:
                        grid_thw = grid.unsqueeze(0)
                    else:
                        grid_thw = grid.reshape(-1, 3)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
            }
        except RuntimeError:
            # Re-raise RuntimeError for missing dependencies
            raise
        except Exception as e:
            logger.error("Error processing qwen item: %s", e)
            logger.debug(traceback.format_exc())
            raise ValueError(
                f"Failed to process Qwen item: {str(e)}. "
                "Check that qwen-vl-utils is installed and image paths are valid."
            ) from e

    def _pad_and_batch(self, inputs_list):
        if not inputs_list:
            return {}

        max_len = max([x["input_ids"].shape[0] for x in inputs_list])

        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        # Reference uses 'padding_side=right' for Qwen3-VL embeddings
        pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else 0
        )

        for x in inputs_list:
            pad_len = max_len - x["input_ids"].shape[0]

            # Right Padding (Reference implementation)
            input_ids = torch.cat(
                [x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
            # Attention Mask: 1 for content, 0 for padding
            attention_mask = torch.cat(
                [x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            if x.get("pixel_values") is not None:
                batch_pixel_values.append(x["pixel_values"])
            if x.get("image_grid_thw") is not None:
                batch_image_grid_thw.append(x["image_grid_thw"])

        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_mask = torch.stack(batch_attention_mask)

        result = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }

        if batch_pixel_values:
            result["pixel_values"] = torch.cat(batch_pixel_values, dim=0)

        if batch_image_grid_thw:
            result["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)

        return result

    def __call__(self, batch):
        query_inputs = []
        pos_inputs = []
        neg_inputs = []

        for item in batch:
            # Query
            q_text = item["query_text"]
            # If query_image is None, it should not affect other fields
            q_img = item.get("query_image", None)

            # If q_text is a string, process it.
            # But if q_text is None and q_img is None, we might skip or produce "NULL"
            # _process_item handles (None, None) -> "NULL"

            q_proc = self._process_item(text=q_text, image=q_img)
            query_inputs.append(q_proc)

            # Positive
            p_img = item["pos_image"]
            p_text = item.get("pos_text", "")

            # If p_img is None (e.g. pure text document?), _process_item handles it
            p_proc = self._process_item(text=p_text if not p_img else None, image=p_img)
            pos_inputs.append(p_proc)

            # Negatives
            if self.mode == "train" and "neg_images" in item:
                current_negs = []
                for n_img in item["neg_images"]:
                    n_proc = self._process_item(image=n_img)
                    current_negs.append(n_proc)
                neg_inputs.append(current_negs)

        q_batch = self._pad_and_batch([q for q in query_inputs if q])
        p_batch = self._pad_and_batch([p for p in pos_inputs if p])

        batch_output = {}

        # Populate Query Keys with Prefix
        if q_batch:
            batch_output["query_input_ids"] = q_batch.get("input_ids")
            batch_output["query_attention_mask"] = q_batch.get("attention_mask")
            batch_output["query_pixel_values"] = q_batch.get("pixel_values")
            batch_output["query_image_grid_thw"] = q_batch.get("image_grid_thw")

            # Legacy/Standard aliases for compatibility if needed
            batch_output["input_ids"] = q_batch.get("input_ids")
            batch_output["attention_mask"] = q_batch.get("attention_mask")

        # Populate Positive Keys with Prefix
        if p_batch:
            batch_output["pos_input_ids"] = p_batch.get("input_ids")
            batch_output["pos_attention_mask"] = p_batch.get("attention_mask")
            batch_output["pos_pixel_values"] = p_batch.get("pixel_values")
            batch_output["pos_image_grid_thw"] = p_batch.get("image_grid_thw")

            # Legacy alias: 'pixel_values' usually means positive image in T2I
            batch_output["pixel_values"] = p_batch.get("pixel_values")

        # Negatives
        if neg_inputs:
            all_neg_procs = []
            neg_counts = []
            for ns in neg_inputs:
                all_neg_procs.extend(ns)
                neg_counts.append(len(ns))

            n_batch = self._pad_and_batch(all_neg_procs)
            batch_output["neg_pixel_values"] = n_batch.get("pixel_values")
            batch_output["neg_image_grid_thw"] = n_batch.get("image_grid_thw")
            batch_output["neg_counts"] = torch.tensor(neg_counts)

        return batch_output
