"""vembed-factory training entrypoint.

Launched by ``accelerate launch`` via :mod:`vembed.cli`.
"""

import argparse
import math
import os
import sys
from typing import Any

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import vembed.data  # noqa: F401 - trigger registry
import vembed.losses  # noqa: F401
import vembed.model  # noqa: F401
from vembed.config import parse_override_args
from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.losses.factory import LossFactory
from vembed.model.modeling import VisualRetrievalModel
from vembed.training.gradient_cache import GradientCache

# Post-init accelerate logger — only use after Accelerator() is created
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vembed-factory training script")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--config_override",
        type=str,
        nargs="*",
        help="Override config keys, e.g. model_name=bert batch_size=32",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    return parser.parse_args()


def _unpack_query_batch(batch: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    if retrieval_mode.startswith("i"):
        if "query_pixel_values" not in batch:
            raise KeyError(
                f"retrieval_mode={retrieval_mode} requires 'query_pixel_values'. "
                f"Got keys: {list(batch.keys())}"
            )
        result: dict[str, Any] = {"pixel_values": batch["query_pixel_values"]}
        # VLMs (e.g. Qwen3-VL) also need input_ids for image items
        if "query_input_ids" in batch:
            result["input_ids"] = batch["query_input_ids"]
            result["attention_mask"] = batch["query_attention_mask"]
        elif "input_ids" in batch:
            result["input_ids"] = batch["input_ids"]
            result["attention_mask"] = batch["attention_mask"]
        if "query_image_grid_thw" in batch and batch["query_image_grid_thw"] is not None:
            result["image_grid_thw"] = batch["query_image_grid_thw"]
        return result

    if retrieval_mode.startswith("m"):
        result = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "query_pixel_values" in batch and batch["query_pixel_values"] is not None:
            result["pixel_values"] = batch["query_pixel_values"]
        if "query_image_grid_thw" in batch and batch["query_image_grid_thw"] is not None:
            result["image_grid_thw"] = batch["query_image_grid_thw"]
        return result

    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


def _unpack_positive_batch(batch: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    if retrieval_mode.endswith("t"):
        return {"input_ids": batch["pos_input_ids"], "attention_mask": batch["pos_attention_mask"]}

    # Image positive: prefer prefixed keys, fallback to legacy aliases
    pv = batch.get("pos_pixel_values")
    if pv is None:
        pv = batch.get("pixel_values")
    result: dict[str, Any] = {"pixel_values": pv}

    # VLMs (e.g. Qwen3-VL) also need input_ids for image items
    if "pos_input_ids" in batch:
        result["input_ids"] = batch["pos_input_ids"]
        result["attention_mask"] = batch["pos_attention_mask"]

    # image_grid_thw for VLMs
    grid = batch.get("pos_image_grid_thw")
    if grid is None:
        grid = batch.get("image_grid_thw")
    if grid is not None:
        result["image_grid_thw"] = grid

    return result


def _maybe_first(embs):
    """Extract a plain tensor from model output (handles ModelOutput, tuples)."""
    if isinstance(embs, torch.Tensor):
        return embs
    if isinstance(embs, tuple):
        return embs[0] if len(embs) > 0 and isinstance(embs[0], torch.Tensor) else embs
    if hasattr(embs, "pooler_output") and embs.pooler_output is not None:
        return embs.pooler_output
    if hasattr(embs, "last_hidden_state") and isinstance(embs.last_hidden_state, torch.Tensor):
        return embs.last_hidden_state[:, 0]
    return embs


def _build_optimizer(model, config):
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(param_groups, lr=float(config["lr"]))


def _resolve_tracker(report_to: str):
    """Map ``report_to`` to Accelerate's ``log_with`` + ``init_kwargs``."""
    if report_to in (None, "none"):
        return None, {}

    init_kwargs: dict[str, Any] = {}

    # swanlab became a built-in tracker in accelerate 1.8.0
    if report_to == "swanlab":
        try:
            import accelerate
            from packaging.version import Version

            if Version(accelerate.__version__) >= Version("1.8.0"):
                return "swanlab", {"swanlab": {"experiment_name": "vembed-factory"}}
        except ImportError:
            pass

        try:
            from swanlab.integration.accelerate import SwanLabTracker

            tracker = SwanLabTracker("vembed-factory")
            return tracker, {}
        except ImportError:
            logger.warning(
                "swanlab requested but neither accelerate>=1.8.0 nor swanlab package found. "
                "Install with: pip install swanlab"
            )
            return None, {}

    return report_to, init_kwargs


def _apply_lora(model, config, accelerator):
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

        # Enable gradient checkpointing if requested
        if config.get("gradient_checkpointing", False):
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
            if hasattr(target, "text_model") and hasattr(
                target.text_model, "gradient_checkpointing_enable"
            ):
                target.text_model.gradient_checkpointing_enable()
                if hasattr(target.text_model, "enable_input_require_grads"):
                    target.text_model.enable_input_require_grads()

            if hasattr(target, "image_model") and hasattr(
                target.image_model, "gradient_checkpointing_enable"
            ):
                target.image_model.gradient_checkpointing_enable()
                if hasattr(target.image_model, "enable_input_require_grads"):
                    target.image_model.enable_input_require_grads()

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


def _load_processor(model_name: str) -> Any:
    """Load processor via the ProcessorRegistry (auto-detect or fallback)."""
    from vembed.model.processors import ProcessorRegistry

    try:
        return ProcessorRegistry.resolve(model_name)
    except Exception as exc:
        logger.warning("ProcessorRegistry.resolve failed for '%s': %s", model_name, exc)
    return None


def main():
    args = parse_args()

    from vembed.config import load_base_config

    config = load_base_config()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    if args.config_override:
        config.update(parse_override_args(args.config_override))

    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True

    os.makedirs(config["output_dir"], exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config.get("ddp_find_unused_parameters", True))
    )
    report_to = config.get("report_to", "none")
    log_with, init_kwargs = _resolve_tracker(report_to)

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with=log_with,
        project_dir=config["output_dir"],
    )
    if log_with is not None:
        accelerator.init_trackers(
            project_name="vembed-factory",
            config=config,
            init_kwargs=init_kwargs,
        )

    accelerator.print(f"Config: {config}")

    model_name = config["model_name"]
    encoder_mode = config.get("encoder_mode", "auto")
    text_model_name = config.get("text_model_name")
    image_model_name = config.get("image_model_name")

    processor = _load_processor(model_name)

    retrieval_mode = config.get("retrieval_mode", "t2i")
    needs_vision = retrieval_mode in ("t2i", "i2i", "i2t", "m2i", "m2t")
    if processor is None and needs_vision and encoder_mode != "composed":
        accelerator.print(
            f"Error: cannot load processor for '{model_name}'. "
            "Install missing tokenizer deps (e.g. sentencepiece) or upgrade transformers."
        )
        sys.exit(1)

    model = VisualRetrievalModel(
        model_name,
        pooling_method=config.get("pooling", "mean"),
        use_mrl=config.get("use_mrl", False),
        mrl_dims=[int(d) for d in config.get("mrl_dims", [768])],
        encoder_mode=encoder_mode,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        attn_implementation=config.get("attn_implementation"),
        torch_dtype=config.get("torch_dtype"),
        projection_dim=config.get("projection_dim"),
        topk_tokens=int(config.get("topk_tokens", 0)),
    )

    if config.get("use_lora", False):
        # When using LoRA with gradient checkpointing, we need to enable input grads
        if config.get("gradient_checkpointing", False):
            pass
        _apply_lora(model, config, accelerator)

    if config.get("use_torch_compile", False) and not config.get("use_fsdp", False):
        accelerator.print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            accelerator.print(f"Warning: torch.compile failed: {e}")

    dataset = VisualRetrievalDataset(
        data_source=config["data_path"],
        processor=processor,
        image_root=config.get("image_root", ""),
        mode="train",
        column_mapping=config.get("column_mapping"),
    )

    collator_kwargs: dict[str, Any] = {
        "processor": processor,
        "mode": "train",
        "prompt": config.get("prompt", "Describe this image."),
    }
    if encoder_mode == "composed":
        from vembed.model.processors import build_image_processor, build_text_processor

        collator_kwargs.update(
            {
                "processor": None,
                "text_processor": build_text_processor(text_model_name),
                "image_processor": build_image_processor(image_model_name),
            }
        )

    collator_cls = CollatorRegistry.get(encoder_mode) or CollatorRegistry.get("default")
    collator = collator_cls(**collator_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
    )

    val_dataloader = None
    if config.get("val_data_path"):
        val_dataset = VisualRetrievalDataset(
            data_source=config["val_data_path"],
            processor=processor,
            image_root=config.get("image_root", ""),
            mode="eval",
            column_mapping=config.get("column_mapping"),
        )
        val_collator = collator_cls(**{**collator_kwargs, "mode": "eval"})
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=True,  # Shuffle for contrastive losses (in-batch negatives)
            collate_fn=val_collator,
            num_workers=4,
            pin_memory=True,
        )

    optimizer = _build_optimizer(model, config)
    num_epochs = int(config["epochs"])
    steps_per_epoch = len(dataloader)
    max_train_steps = num_epochs * steps_per_epoch

    warmup_steps = int(config.get("warmup_steps", 0))
    if warmup_steps == 0:
        warmup_steps = math.ceil(max_train_steps * float(config.get("warmup_ratio", 0.1)))

    lr_scheduler = get_scheduler(
        name=config.get("scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    accelerator.print(
        f"Scheduler: {config.get('scheduler_type', 'cosine')}, "
        f"warmup={warmup_steps}, total={max_train_steps}"
    )

    criterion = LossFactory.create(config)

    teacher_model = None
    distillation_loss_fn = None
    distillation_alpha = float(config.get("distillation_alpha", 0.5))

    if config.get("teacher_model_name"):
        accelerator.print(f"Loading teacher: {config['teacher_model_name']}")
        teacher_model = VisualRetrievalModel(
            config["teacher_model_name"],
            pooling_method=config.get("pooling", "mean"),
            use_mrl=False,
            encoder_mode=encoder_mode,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            attn_implementation=config.get("attn_implementation"),
            torch_dtype=config.get("torch_dtype"),
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        teacher_model = accelerator.prepare(teacher_model)
        distillation_loss_fn = LossFactory.create_distillation_loss(config)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    grad_cache = GradientCache(
        loss_fn=criterion,
        chunk_size=config["gradient_cache_chunk_size"],
        accelerator=accelerator,
        retrieval_mode=config.get("retrieval_mode", "t2i"),
    )

    save_steps = int(config.get("save_steps", 0) or 0)
    logging_steps = int(config.get("logging_steps", 10))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    def save_checkpoint(path: str):
        if not accelerator.is_local_main_process:
            return
        accelerator.save_state(path)
        accelerator.unwrap_model(model).save_pretrained(path)
        if processor:
            processor.save_pretrained(path)
        # Persist vembed-specific config (topk_tokens, pooling, etc.)
        import json

        vembed_cfg = {
            "pooling_method": config.get("pooling", "mean"),
            "projection_dim": config.get("projection_dim"),
            "topk_tokens": int(config.get("topk_tokens", 0)),
            "retrieval_mode": config.get("retrieval_mode", "t2i"),
            "loss_type": config.get("loss_type", "infonce"),
            "use_mrl": config.get("use_mrl", False),
            "mrl_dims": config.get("mrl_dims"),
        }
        cfg_path = os.path.join(path, "vembed_config.json")
        with open(cfg_path, "w") as fp:
            json.dump(vembed_cfg, fp, indent=2)
        accelerator.print(f"Saved vembed_config.json → {cfg_path}")

    def evaluate() -> float:
        model.eval()
        total_loss, num_batches = 0.0, 0
        accelerator.print("\nRunning validation...")

        with torch.no_grad():
            for batch in tqdm(val_dataloader, disable=not accelerator.is_local_main_process):
                q_embs = _maybe_first(model(**_unpack_query_batch(batch, retrieval_mode)))
                p_embs = _maybe_first(model(**_unpack_positive_batch(batch, retrieval_mode)))

                loss_kwargs = {}
                if "labels" in batch:
                    loss_kwargs["labels"] = batch["labels"]

                total_loss += criterion(q_embs, p_embs, None, **loss_kwargs).item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accelerator.print(f"Validation loss: {avg_loss:.4f}\n")
        if log_with is not None:
            accelerator.log({"val/loss": avg_loss}, step=global_step)
        model.train()
        return avg_loss

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(
            tqdm(dataloader, disable=not accelerator.is_local_main_process)
        ):
            global_step += 1

            if config.get("use_gradient_cache", False):
                loss_val = grad_cache.step(model, batch)
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                q_embs = _maybe_first(model(**_unpack_query_batch(batch, retrieval_mode)))
                p_embs = _maybe_first(model(**_unpack_positive_batch(batch, retrieval_mode)))

                n_embs = None
                if batch.get("neg_pixel_values") is not None:
                    neg_kwargs: dict[str, Any] = {"pixel_values": batch["neg_pixel_values"]}
                    if "neg_input_ids" in batch:
                        neg_kwargs["input_ids"] = batch["neg_input_ids"]
                        neg_kwargs["attention_mask"] = batch["neg_attention_mask"]
                    if batch.get("neg_image_grid_thw") is not None:
                        neg_kwargs["image_grid_thw"] = batch["neg_image_grid_thw"]
                    n_embs = _maybe_first(model(**neg_kwargs))

                loss_kwargs = {}
                if "labels" in batch:
                    loss_kwargs["labels"] = batch["labels"]

                loss = criterion(q_embs, p_embs, n_embs, **loss_kwargs)

                if teacher_model is not None and distillation_loss_fn is not None:
                    with torch.no_grad():
                        t_q = _maybe_first(
                            teacher_model(**_unpack_query_batch(batch, retrieval_mode))
                        )
                        t_p = _maybe_first(
                            teacher_model(**_unpack_positive_batch(batch, retrieval_mode))
                        )
                    distill_loss = distillation_loss_fn(q_embs, p_embs, t_q, t_p)
                    loss = distillation_alpha * loss + (1.0 - distillation_alpha) * distill_loss

                accelerator.backward(loss)
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_val = loss.item()

            if global_step % logging_steps == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.print(
                    f"  step {global_step} | loss={loss_val:.4f} | lr={current_lr:.2e}"
                )
                if log_with is not None:
                    accelerator.log(
                        {
                            "train/loss": loss_val,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + (step + 1) / steps_per_epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

            if save_steps > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    os.path.join(config["output_dir"], f"checkpoint-step-{global_step}")
                )

        save_checkpoint(os.path.join(config["output_dir"], f"checkpoint-epoch-{epoch + 1}"))

        if val_dataloader:
            evaluate()

    if log_with is not None:
        accelerator.end_training()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
