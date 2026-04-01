"""vembed-factory training entrypoint.

Launched by ``accelerate launch`` via :mod:`vembed.cli`.

This module orchestrates the training pipeline by composing modularized
components from the training package.
"""

import os
import sys
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.distributed.algorithms.ddp_comm_hooks"
)

from accelerate import Accelerator, DistributedDataParallelKwargs  # noqa: E402
from accelerate.logging import get_logger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import vembed.data  # noqa: F401, E402 - trigger registry
import vembed.losses  # noqa: F401, E402
import vembed.model  # noqa: F401, E402
from vembed.data.dataset import VisualRetrievalDataset  # noqa: E402
from vembed.data.registry import CollatorRegistry  # noqa: E402
from vembed.data.validation import print_validation_report, validate_dataset  # noqa: E402
from vembed.losses.factory import LossFactory  # noqa: E402
from vembed.training.config import (  # noqa: E402
    get_distributed_config,
    load_and_parse_config,
    prepare_output_dir,
)
from vembed.training.evaluator import Evaluator  # noqa: E402
from vembed.training.model_builder import (  # noqa: E402
    _log_fsdp_param_summary,
    apply_lora,
    build_model,
    build_teacher_model,
    compile_model,
    enable_static_graph,
    load_processor,
    unify_model_dtype_for_fsdp,
    validate_processor,
)
from vembed.training.optimizer_builder import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    resolve_tracker,
)
from vembed.training.checkpoint import load_checkpoint  # noqa: E402
from vembed.training.training_loop import Trainer  # noqa: E402

# Post-init accelerate logger — only use after Accelerator() is created
logger = get_logger(__name__)


def main():
    """Main training entrypoint."""
    # Load and merge configuration
    config = load_and_parse_config()
    prepare_output_dir(config)

    # Note: GPU memory limit is set in cli.py before accelerator initialization
    # Do NOT call torch.cuda.set_per_process_memory_fraction here (can only be called once per process)

    use_grad_checkpointing, use_gradient_cache, find_unused = get_distributed_config(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
    report_to = config.get("report_to", "none")
    run_name = config.get("run_name")
    run_tags = config.get("run_tags")
    run_notes = config.get("run_notes")
    log_with, init_kwargs = resolve_tracker(
        report_to,
        run_name=run_name,
        run_tags=run_tags,
        run_notes=run_notes,
    )

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with=log_with,
        project_dir=config["output_dir"],
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if log_with is not None:
        accelerator.init_trackers(
            project_name="vembed-factory",
            config=config,
            init_kwargs=init_kwargs,
        )

    accelerator.print("\n" + "=" * 70)
    accelerator.print("Training Configuration")
    accelerator.print("=" * 70)
    accelerator.print(f"Model: {config['model_name']}")
    accelerator.print(f"Data: {config['data_path']}")
    accelerator.print(f"Output: {config['output_dir']}")
    accelerator.print("=" * 70 + "\n")

    # Build model and processor
    processor = load_processor(config["model_name"])
    retrieval_mode = config.get("retrieval_mode", "t2i")
    needs_vision = retrieval_mode in ("t2i", "i2i", "i2t", "m2i", "m2t")
    encoder_mode = config.get("encoder_mode", "auto")
    validate_processor(processor, needs_vision, config["model_name"], accelerator)

    model = build_model(config)

    if config.get("use_lora", False):
        apply_lora(model, config, accelerator)

    model = compile_model(model, config, accelerator)

    # Build teacher model for distillation if configured
    teacher_model = build_teacher_model(config)
    distillation_loss_fn = None
    if teacher_model is not None:
        accelerator.print(f"Loading teacher: {config['teacher_model_name']}")
        teacher_model = accelerator.prepare(teacher_model)
        distillation_loss_fn = LossFactory.create_distillation_loss(config)

    # Validate dataset if configured
    if config.get("validate_data", False):
        accelerator.print("Validating training dataset...")
        stats = validate_dataset(
            data_source=config["data_path"],
            column_mapping=config.get("column_mapping"),
            sample_size=100,
            skip_invalid=config.get("skip_invalid_records", True),
        )
        if accelerator.is_main_process:
            print_validation_report(stats)

    # Prepare dataset and dataloader
    with accelerator.main_process_first():
        dataset = VisualRetrievalDataset(
            data_source=config["data_path"],
            processor=processor,
            image_root=config.get("image_root", ""),
            mode="train",
            column_mapping=config.get("column_mapping"),
            enable_image_cache=config.get("enable_image_cache", False),
            auto_clean=False,
            validate_columns=False,
        )

    collator_kwargs: dict = {
        "processor": processor,
        "mode": "train",
        "retrieval_mode": retrieval_mode,
        "prompt": config.get("prompt", "Describe this image."),
    }
    if encoder_mode == "composed":
        from vembed.model.processors import build_image_processor, build_text_processor

        collator_kwargs.update(
            {
                "processor": None,
                "text_processor": build_text_processor(config.get("text_model_name")),
                "image_processor": build_image_processor(config.get("image_model_name")),
            }
        )

    # Select collator by encoder_mode (model family)
    collator_cls = (
        CollatorRegistry.get(encoder_mode)
        # retrieval_mode will be used by collator internally
        or CollatorRegistry.get("clip")  # New default: CLIP-family collator
    )
    collator = collator_cls(**collator_kwargs)

    # Configure DataLoader with configurable parameters
    num_workers = config.get("num_workers", 4)
    pin_memory = config.get("pin_memory", True)
    prefetch_factor = config.get("prefetch_factor", 2)
    persistent_workers = num_workers > 0 and config.get("persistent_workers", True)

    # Optimization: Auto-optimize prefetch_factor if using default value
    if prefetch_factor == 2 and num_workers > 0:
        batch_size = config["batch_size"]
        # Adaptive formula: prefetch_factor = min(8, max(2, 512 // batch_size))
        auto_prefetch = max(2, min(8, 512 // batch_size))
        prefetch_factor = auto_prefetch
        accelerator.print(
            f"Auto-optimized prefetch_factor: {prefetch_factor} "
            f"(based on batch_size={batch_size}, num_workers={num_workers})"
        )

    dataloader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Add advanced options if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": persistent_workers,
            }
        )

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Prepare validation dataloader if configured
    val_dataloader = None
    if config.get("val_data_path"):
        # Validate validation dataset if configured
        if config.get("validate_data", False):
            accelerator.print("Validating validation dataset...")
            val_stats = validate_dataset(
                data_source=config["val_data_path"],
                column_mapping=config.get("column_mapping"),
                sample_size=100,
                skip_invalid=config.get("skip_invalid_records", True),
            )
            if accelerator.is_main_process:
                accelerator.print("Validation Dataset Report:")
                print_validation_report(val_stats)

        with accelerator.main_process_first():
            val_dataset = VisualRetrievalDataset(
                data_source=config["val_data_path"],
                processor=processor,
                image_root=config.get("image_root", ""),
                mode="eval",
                column_mapping=config.get("column_mapping"),
                enable_image_cache=False,  # Optimization: Disable cache for val (single pass)
                auto_clean=False,
                validate_columns=False,
            )
        val_collator = collator_cls(**{**collator_kwargs, "mode": "eval"})
        eval_batch_size = config.get("eval_batch_size", config["batch_size"])

        # Validation dataloader with optimized settings
        val_dataloader_kwargs = {
            "batch_size": eval_batch_size,
            "shuffle": False,  # No need to shuffle validation data
            "collate_fn": val_collator,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        if num_workers > 0:
            val_dataloader_kwargs.update(
                {
                    "prefetch_factor": max(1, prefetch_factor // 2),  # Smaller prefetch for eval
                    "persistent_workers": persistent_workers,
                }
            )

        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # Build loss function
    criterion = LossFactory.create(config)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config, criterion=criterion)
    num_epochs = int(config["epochs"])
    steps_per_epoch = len(dataloader)

    scheduler, warmup_steps = build_scheduler(optimizer, config, num_epochs, steps_per_epoch)

    max_train_steps = num_epochs * steps_per_epoch
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    effective_batch_size = config["batch_size"] * grad_accum_steps

    accelerator.print("\n" + "-" * 70)
    accelerator.print("Training Setup")
    accelerator.print("-" * 70)
    accelerator.print(
        f"Epochs: {num_epochs} | Steps/Epoch: {steps_per_epoch} | Total Steps: {max_train_steps}"
    )
    accelerator.print(
        f"Batch Size: {config['batch_size']} | Gradient Accumulation: {grad_accum_steps}x | Effective: {effective_batch_size}"
    )
    accelerator.print(
        f"Learning Rate: {config.get('learning_rate', 'auto')} | Scheduler: {config.get('scheduler_type', 'cosine')}"
    )
    accelerator.print(f"Warmup Steps: {warmup_steps}")

    if config.get("eval_steps", 0) > 0:
        accelerator.print(f"Evaluation: every {config['eval_steps']} steps")
    if config.get("early_stopping_patience", -1) > 0:
        accelerator.print(
            f"Early Stopping: patience={config['early_stopping_patience']}, metric={config.get('eval_metric', 'val/loss')}"
        )
    accelerator.print("-" * 70 + "\n")

    # Unify dtype before FSDP wrapping to avoid "mixed dtype" errors during all_gather
    unify_model_dtype_for_fsdp(model, config, accelerator)
    _log_fsdp_param_summary(model, accelerator)

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Log distributed training setup
    accelerator.print("\n" + "-" * 70)
    accelerator.print("Distributed Training")
    accelerator.print("-" * 70)
    if config.get("use_fsdp"):
        accelerator.print("SUCCESS: FSDP (Fully Sharded Data Parallel) enabled")
    if config.get("use_gradient_cache"):
        chunk_size = config.get("gradient_cache_chunk_size", config["batch_size"])
        accelerator.print(f"SUCCESS: Gradient Cache enabled (chunk_size={chunk_size})")
    if config.get("use_fsdp") and config.get("use_gradient_cache"):
        accelerator.print("SUCCESS: FSDP + Gradient Cache: memory-efficient large model training")
        accelerator.print("  WARNING: Note: no_sync optimization disabled for FSDP safety")
    if config.get("use_lora"):
        accelerator.print(f"SUCCESS: LoRA enabled (r={config.get('lora_r', 'auto')})")
    accelerator.print(
        f"Processes: {accelerator.num_processes} | Gradient Sync: every {grad_accum_steps} steps"
    )
    accelerator.print("-" * 70 + "\n")

    # Enable static graph for DDP optimization
    enable_static_graph(model, config, accelerator)

    # Store processor in config for trainer
    config["processor"] = processor

    # Create evaluator if validation dataloader exists
    evaluator = None
    if val_dataloader:
        evaluator = Evaluator(
            model=model,
            criterion=criterion,
            accelerator=accelerator,
            retrieval_mode=retrieval_mode,
            log_with=log_with,
        )

    # Load checkpoint and resume if specified
    resume_state = None
    if config.get("resume_from_checkpoint"):
        resume_path = config["resume_from_checkpoint"]
        logger.info(f"Resuming from checkpoint: {resume_path}")

        try:
            resume_state = load_checkpoint(
                resume_path,
                model,
                accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                processor=processor,
                mode=config.get("resume_mode", "full"),
            )
            logger.info(f"Checkpoint loaded successfully: {resume_state}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise

    # Create and run trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        criterion=criterion,
        accelerator=accelerator,
        config=config,
        scheduler=scheduler,
        teacher_model=teacher_model,
        distillation_loss_fn=distillation_loss_fn,
        evaluator=evaluator,
        val_dataloader=val_dataloader,
        training_state=resume_state,
    )

    trainer.train()

    # Cleanup
    if log_with is not None:
        accelerator.end_training()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
