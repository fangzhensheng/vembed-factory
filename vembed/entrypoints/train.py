"""vembed-factory training entrypoint.

Launched by ``accelerate launch`` via :mod:`vembed.cli`.

This module orchestrates the training pipeline by composing modularized
components from the training package.
"""

import os
import sys

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from transformers import get_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import vembed.data  # noqa: F401 - trigger registry
import vembed.losses  # noqa: F401
import vembed.model  # noqa: F401
from vembed.config import load_base_config, parse_override_args
from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.losses.factory import LossFactory
from vembed.training.checkpoint import save_checkpoint
from vembed.training.config import (
    get_distributed_config,
    load_and_parse_config,
    prepare_output_dir,
)
from vembed.training.evaluator import Evaluator
from vembed.training.model_builder import (
    apply_lora,
    build_model,
    build_teacher_model,
    compile_model,
    enable_static_graph,
    load_processor,
    validate_processor,
)
from vembed.training.optimizer_builder import build_optimizer, build_scheduler, resolve_tracker
from vembed.training.training_loop import Trainer

# Post-init accelerate logger — only use after Accelerator() is created
logger = get_logger(__name__)


def main():
    """Main training entrypoint."""
    # Load and merge configuration
    config = load_and_parse_config()
    prepare_output_dir(config)

    # Initialize distributed training
    use_grad_checkpointing, use_gradient_cache, find_unused = get_distributed_config(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
    report_to = config.get("report_to", "none")
    log_with, init_kwargs = resolve_tracker(report_to)

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

    # Prepare dataset and dataloader
    dataset = VisualRetrievalDataset(
        data_source=config["data_path"],
        processor=processor,
        image_root=config.get("image_root", ""),
        mode="train",
        column_mapping=config.get("column_mapping"),
    )

    collator_kwargs: dict = {
        "processor": processor,
        "mode": "train",
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

    # Prepare validation dataloader if configured
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
            shuffle=True,
            collate_fn=val_collator,
            num_workers=4,
            pin_memory=True,
        )

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    num_epochs = int(config["epochs"])
    steps_per_epoch = len(dataloader)

    scheduler, warmup_steps = build_scheduler(optimizer, config, num_epochs, steps_per_epoch)

    max_train_steps = num_epochs * steps_per_epoch
    accelerator.print(
        f"Scheduler: {config.get('scheduler_type', 'cosine')}, "
        f"warmup={warmup_steps}, total={max_train_steps}"
    )

    # Build loss function
    criterion = LossFactory.create(config)

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

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
    )

    trainer.train()

    # Cleanup
    if log_with is not None:
        accelerator.end_training()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
