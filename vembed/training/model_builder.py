"""Model initialization and setup utilities."""

import sys
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry

logger = get_logger(__name__)


def load_processor(model_name: str) -> Any | None:
    """Load processor via the ProcessorRegistry.

    Args:
        model_name: Name of the model to load processor for.

    Returns:
        Loaded processor, or None if not found or error occurs.
    """
    try:
        return ProcessorRegistry.resolve(model_name)
    except Exception as exc:
        logger.warning("ProcessorRegistry.resolve failed for '%s': %s", model_name, exc)
    return None


def validate_processor(
    processor: Any | None,
    needs_vision: bool,
    model_name: str,
    accelerator: Accelerator,
) -> None:
    """Validate that processor is available when needed.

    Args:
        processor: Loaded processor or None.
        needs_vision: Whether vision processing is required.
        model_name: Model name for error messages.
        accelerator: Accelerate instance for printing.

    Raises:
        SystemExit: If processor is required but not available.
    """
    if processor is None and needs_vision:
        accelerator.print(
            f"Error: cannot load processor for '{model_name}'. "
            "Install missing tokenizer deps (e.g. sentencepiece) or upgrade transformers."
        )
        sys.exit(1)


def build_model(config: dict[str, Any]) -> VisualRetrievalModel:
    """Build the visual retrieval model.

    Args:
        config: Configuration dict with model parameters.

    Returns:
        Initialized VisualRetrievalModel.
    """
    model = VisualRetrievalModel(
        config["model_name"],
        pooling_method=config.get("pooling_method"),
        use_mrl=config.get("use_mrl", False),
        mrl_dims=[int(d) for d in config.get("mrl_dims", [768])],
        encoder_mode=config.get("encoder_mode", "auto"),
        text_model_name=config.get("text_model_name"),
        image_model_name=config.get("image_model_name"),
        attn_implementation=config.get("attn_implementation"),
        torch_dtype=config.get("torch_dtype"),
        projection_dim=config.get("projection_dim"),
        topk_tokens=int(config.get("topk_tokens", 0)),
    )
    return model


def build_teacher_model(config: dict[str, Any]) -> VisualRetrievalModel | None:
    """Build teacher model for knowledge distillation if configured.

    Args:
        config: Configuration dict.

    Returns:
        Initialized teacher model, or None if not configured.
    """
    if not config.get("teacher_model_name"):
        return None

    teacher_model = VisualRetrievalModel(
        config["teacher_model_name"],
        pooling_method=config.get("pooling_method"),
        use_mrl=False,
        encoder_mode=config.get("encoder_mode", "auto"),
        text_model_name=config.get("text_model_name"),
        image_model_name=config.get("image_model_name"),
        attn_implementation=config.get("attn_implementation"),
        torch_dtype=config.get("torch_dtype"),
    )
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    return teacher_model


def apply_lora(model: VisualRetrievalModel, config: dict[str, Any], accelerator: Accelerator) -> None:
    """Apply LoRA to model parameters.

    Args:
        model: Model to apply LoRA to.
        config: Configuration dict with LoRA parameters.
        accelerator: Accelerate instance for printing and device handling.

    Raises:
        SystemExit: If peft library is not installed or LoRA application fails.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        accelerator.print("Error: 'peft' library not found. Install with: pip install peft")
        sys.exit(1)

    accelerator.print("Applying LoRA")

    lora_cfg = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("lora_target_modules")
        or ["q_proj", "v_proj", "query", "value", "key", "dense"],
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        modules_to_save=["classifier", "pooler", "projector"],
    )

    try:
        target = model.backend

        if config.get("gradient_checkpointing", False):
            _enable_gradient_checkpointing(target, accelerator)

        if hasattr(target, "backbone"):
            target.backbone = get_peft_model(target.backbone, lora_cfg)
            target.backbone.print_trainable_parameters()
        else:
            model.backend = get_peft_model(target, lora_cfg)
            model.backend.print_trainable_parameters()
        accelerator.print("LoRA injected")
    except Exception as exc:
        accelerator.print(f"Error applying LoRA: {exc}")
        sys.exit(1)


def _enable_gradient_checkpointing(target: torch.nn.Module, accelerator: Accelerator) -> None:
    """Enable gradient checkpointing on model.

    Args:
        target: Model or backbone to enable checkpointing on.
        accelerator: Accelerate instance for printing.
    """
    accelerator.print("Enabling gradient checkpointing...")

    if hasattr(target, "backbone"):
        target.backbone.gradient_checkpointing_enable()
        if hasattr(target.backbone, "enable_input_require_grads"):
            target.backbone.enable_input_require_grads()
    elif hasattr(target, "gradient_checkpointing_enable"):
        target.gradient_checkpointing_enable()
        if hasattr(target, "enable_input_require_grads"):
            target.enable_input_require_grads()

    # Handle composed models (text_model / image_model)
    if hasattr(target, "text_model") and hasattr(target.text_model, "gradient_checkpointing_enable"):
        target.text_model.gradient_checkpointing_enable()
        if hasattr(target.text_model, "enable_input_require_grads"):
            target.text_model.enable_input_require_grads()

    if hasattr(target, "image_model") and hasattr(target.image_model, "gradient_checkpointing_enable"):
        target.image_model.gradient_checkpointing_enable()
        if hasattr(target.image_model, "enable_input_require_grads"):
            target.image_model.enable_input_require_grads()


def unify_model_dtype_for_fsdp(model: torch.nn.Module, config: dict[str, Any], accelerator: Accelerator) -> None:
    """Ensure all model parameters have uniform dtype for FSDP compatibility.

    FSDP requires all parameters to have the same dtype. This function converts
    all parameters to match the configured torch_dtype.

    Args:
        model: Model to unify dtype on.
        config: Configuration dict with 'torch_dtype'.
        accelerator: Accelerate instance for printing.
    """
    if not config.get("use_fsdp", False):
        return

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    target_dtype_str = config.get("torch_dtype", "bfloat16")
    target_dtype = dtype_map.get(str(target_dtype_str), torch.bfloat16)

    dtypes_found = {param.dtype for param in model.parameters()}

    if len(dtypes_found) > 1:
        dtype_counts = {}
        for param in model.parameters():
            dtype_str = str(param.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + param.numel()

        accelerator.print(f"⚠️  FSDP dtype mismatch detected: {dtype_counts}")
        accelerator.print(f"   Converting all parameters to {target_dtype}...")

        for param in model.parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)

        accelerator.print(f"✓ All parameters converted to {target_dtype}")
    else:
        accelerator.print(f"✓ FSDP dtype check: all parameters already in {list(dtypes_found)[0]}")


def _log_fsdp_param_summary(model: torch.nn.Module, accelerator: Accelerator) -> None:
    """Log model parameter summary for FSDP debugging."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"FSDP parameter summary:")
    accelerator.print(f"  Total: {total_params:,} ({total_params/1e9:.2f}B)")
    accelerator.print(f"  Trainable: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    accelerator.print(f"  Frozen: {total_params - trainable_params:,}")


def compile_model(model: torch.nn.Module, config: dict[str, Any], accelerator: Accelerator) -> torch.nn.Module:
    """Optionally compile model with torch.compile.

    Args:
        model: Model to compile.
        config: Configuration dict with 'use_torch_compile' and 'use_fsdp' flags.
        accelerator: Accelerate instance for printing.

    Returns:
        Compiled model or original model if compilation is disabled/unsupported.
    """
    if config.get("use_torch_compile", False) and not config.get("use_fsdp", False):
        accelerator.print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            accelerator.print(f"Warning: torch.compile failed: {e}")

    return model


def enable_static_graph(model: torch.nn.Module, config: dict[str, Any], accelerator: Accelerator) -> None:
    """Enable static graph for DDP optimization.

    Args:
        model: Model to enable static graph on.
        config: Configuration dict with encoder_mode, gradient checkpointing, gradient cache flags.
        accelerator: Accelerate instance for printing.

    Note:
        Static graph is skipped if using gradient_cache or gradient_checkpointing
        since they modify the graph structure.
    """
    encoder_mode = config.get("encoder_mode", "auto")
    use_gradient_cache = config.get("use_gradient_cache", False)
    use_grad_checkpointing = config.get("gradient_checkpointing", False)

    if (encoder_mode != "composed" and
        not use_gradient_cache and
        not use_grad_checkpointing and
        hasattr(model, "_set_static_graph")):
        try:
            model._set_static_graph()
            accelerator.print("Enabled static graph for DDP")
        except Exception as e:
            accelerator.print(f"Warning: Could not set static graph: {e}")
