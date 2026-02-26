"""Core training loop implementation."""

import os
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

from vembed.core.gradient_cache import GradientCache
from vembed.training.checkpoint import save_checkpoint
from vembed.training.data_utils import (
    concat_batches,
    maybe_first,
    unpack_negative_batch,
    unpack_positive_batch,
    unpack_query_batch,
)

logger = get_logger(__name__)


class Trainer:
    """Orchestrates the training loop with support for distributed training, gradient caching, and distillation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: Any,
        criterion: Any,
        accelerator: Accelerator,
        config: dict[str, Any],
        scheduler: Any = None,
        teacher_model: torch.nn.Module | None = None,
        distillation_loss_fn: Any = None,
        evaluator: Any = None,
        val_dataloader: Any = None,
    ):
        """Initialize trainer.

        Args:
            model: Student model to train.
            optimizer: Optimizer.
            dataloader: Training dataloader.
            criterion: Loss function.
            accelerator: Accelerate instance for distributed training.
            config: Configuration dict.
            scheduler: Learning rate scheduler.
            teacher_model: Optional teacher model for knowledge distillation.
            distillation_loss_fn: Optional distillation loss function.
            evaluator: Optional evaluator for validation.
            val_dataloader: Optional validation dataloader.
        """
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.criterion = criterion
        self.accelerator = accelerator
        self.config = config
        self.scheduler = scheduler
        self.teacher_model = teacher_model
        self.distillation_loss_fn = distillation_loss_fn
        self.evaluator = evaluator
        self.val_dataloader = val_dataloader

        # Training configuration
        self.num_epochs = int(config["epochs"])
        self.logging_steps = int(config.get("logging_steps", 10))
        self.save_steps = int(config.get("save_steps", 0) or 0)
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.distillation_alpha = float(config.get("distillation_alpha", 0.5))
        self.use_gradient_cache = config.get("use_gradient_cache", False)
        self.retrieval_mode = config.get("retrieval_mode", "t2i")

        # Gradient cache setup
        if self.use_gradient_cache:
            self.grad_cache = GradientCache(
                loss_fn=criterion,
                chunk_size=config["gradient_cache_chunk_size"],
                accelerator=accelerator,
                retrieval_mode=self.retrieval_mode,
            )
        else:
            self.grad_cache = None

        # Get processor and model config for batch concatenation
        self.processor = config.get("processor")
        self.encoder_mode = config.get("encoder_mode", "auto")

    def train(self) -> None:
        """Run the complete training loop."""
        self.model.train()
        global_step = 0
        steps_per_epoch = len(self.dataloader)

        for epoch in range(self.num_epochs):
            self.accelerator.print(f"Epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(
                tqdm(self.dataloader, disable=not self.accelerator.is_local_main_process)
            ):
                global_step += 1

                loss_val = self._train_step(batch)

                if global_step % self.logging_steps == 0:
                    self._log_step(global_step, loss_val, epoch, step, steps_per_epoch)

                if self.save_steps > 0 and global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

            # Save checkpoint after each epoch
            self._save_checkpoint_epoch(epoch)

            # Validate if dataloader is provided
            if self.val_dataloader:
                avg_loss = self.evaluator.evaluate(self.val_dataloader, global_step)
                if self.accelerator.log_with is not None:
                    self.accelerator.log({"val/loss": avg_loss}, step=global_step)

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Execute a single training step.

        Args:
            batch: Input batch.

        Returns:
            Loss value for the step.
        """
        if self.use_gradient_cache:
            return self._step_with_gradient_cache(batch)
        else:
            return self._step_standard(batch)

    def _step_with_gradient_cache(self, batch: dict[str, Any]) -> float:
        """Training step using gradient cache for memory efficiency.

        Args:
            batch: Input batch.

        Returns:
            Loss value.
        """
        loss_val = self.grad_cache.step(self.model, batch)
        if self.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss_val

    def _step_standard(self, batch: dict[str, Any]) -> float:
        """Standard training step.

        Args:
            batch: Input batch.

        Returns:
            Loss value.
        """
        q_inputs = unpack_query_batch(batch, self.retrieval_mode)
        p_inputs = unpack_positive_batch(batch, self.retrieval_mode)
        n_inputs = unpack_negative_batch(batch)

        should_concat = self._should_concat_inputs(q_inputs, p_inputs)

        if should_concat:
            q_embs, p_embs, n_embs = self._forward_concatenated(
                q_inputs, p_inputs, n_inputs
            )
        else:
            q_embs = maybe_first(self.model(**q_inputs))
            p_embs = maybe_first(self.model(**p_inputs))
            n_embs = maybe_first(self.model(**n_inputs)) if n_inputs else None

        loss_kwargs = {}
        if "labels" in batch:
            loss_kwargs["labels"] = batch["labels"]

        loss = self.criterion(q_embs, p_embs, n_embs, **loss_kwargs)

        # Knowledge distillation
        if self.teacher_model is not None and self.distillation_loss_fn is not None:
            loss = self._apply_distillation(batch, q_embs, p_embs, loss)

        self.accelerator.backward(loss)
        if self.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def _should_concat_inputs(self, q_inputs: dict[str, Any], p_inputs: dict[str, Any]) -> bool:
        """Determine if inputs should be concatenated.

        Concatenation is used for unified models (e.g., Qwen-VL) that tokenize
        both text and images. Composed models (e.g., CLIP) use separate inputs.

        Args:
            q_inputs: Query inputs.
            p_inputs: Positive inputs.

        Returns:
            True if inputs should be concatenated.
        """
        if self.encoder_mode == "composed":
            return False

        q_keys = set(q_inputs.keys())
        p_keys = set(p_inputs.keys())
        # If both have input_ids, it's likely a unified LLM/VLM
        return "input_ids" in q_keys and "input_ids" in p_keys

    def _forward_concatenated(
        self,
        q_inputs: dict[str, Any],
        p_inputs: dict[str, Any],
        n_inputs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass with concatenated inputs.

        Args:
            q_inputs: Query inputs.
            p_inputs: Positive inputs.
            n_inputs: Optional negative inputs.

        Returns:
            Tuple of (query_embs, positive_embs, negative_embs or None).
        """
        batches_to_concat = [q_inputs, p_inputs]
        if n_inputs:
            batches_to_concat.append(n_inputs)

        pad_id = self._get_pad_token_id()
        concatenated_inputs, batch_sizes = concat_batches(batches_to_concat, pad_token_id=pad_id)

        # Use no_sync() to avoid DDP parameter ready-multiple times error
        # When concat [q, p, n], parameters are used 3x in same forward
        use_no_sync = bool(self.accelerator and self.accelerator.num_processes > 1)
        if use_no_sync and hasattr(self.model, "no_sync"):
            with self.model.no_sync():
                all_embs = maybe_first(self.model(**concatenated_inputs))
        else:
            all_embs = maybe_first(self.model(**concatenated_inputs))

        # Split concatenated output back into q, p, n embeddings
        q_embs = all_embs[: batch_sizes[0]]
        p_embs = all_embs[batch_sizes[0] : batch_sizes[0] + batch_sizes[1]]
        n_embs = all_embs[batch_sizes[0] + batch_sizes[1] :] if n_inputs else None

        return q_embs, p_embs, n_embs

    def _get_pad_token_id(self) -> int:
        """Get pad token ID from processor or model config.

        Returns:
            Pad token ID (default: 0).
        """
        if self.processor and hasattr(self.processor, "tokenizer"):
            if self.processor.tokenizer.pad_token_id is not None:
                return self.processor.tokenizer.pad_token_id
        if hasattr(self.model, "config") and hasattr(self.model.config, "pad_token_id"):
            if self.model.config.pad_token_id is not None:
                return self.model.config.pad_token_id
        return 0

    def _apply_distillation(
        self,
        batch: dict[str, Any],
        q_embs: torch.Tensor,
        p_embs: torch.Tensor,
        student_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Apply knowledge distillation loss.

        Args:
            batch: Input batch.
            q_embs: Student query embeddings.
            p_embs: Student positive embeddings.
            student_loss: Student loss.

        Returns:
            Combined student + distillation loss.
        """
        with torch.no_grad():
            t_q = maybe_first(
                self.teacher_model(**unpack_query_batch(batch, self.retrieval_mode))
            )
            t_p = maybe_first(
                self.teacher_model(**unpack_positive_batch(batch, self.retrieval_mode))
            )

        distill_loss = self.distillation_loss_fn(q_embs, p_embs, t_q, t_p)
        loss = self.distillation_alpha * student_loss + (1.0 - self.distillation_alpha) * distill_loss
        return loss

    def _log_step(
        self,
        global_step: int,
        loss_val: float,
        epoch: int,
        step: int,
        steps_per_epoch: int,
    ) -> None:
        """Log training metrics.

        Args:
            global_step: Current global step.
            loss_val: Loss value.
            epoch: Current epoch.
            step: Step within epoch.
            steps_per_epoch: Total steps per epoch.
        """
        current_lr = self.scheduler.get_last_lr()[0]
        self.accelerator.print(
            f"  step {global_step} | loss={loss_val:.4f} | lr={current_lr:.2e}"
        )
        if self.accelerator.log_with is not None:
            self.accelerator.log(
                {
                    "train/loss": loss_val,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + (step + 1) / steps_per_epoch,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

    def _save_checkpoint(self, global_step: int) -> None:
        """Save checkpoint at a specific step.

        Args:
            global_step: Current global step.
        """
        checkpoint_dir = os.path.join(
            self.config["output_dir"], f"checkpoint-step-{global_step}"
        )
        save_checkpoint(
            checkpoint_dir,
            self.model,
            self.accelerator,
            processor=self.processor,
            config=self.config,
        )

    def _save_checkpoint_epoch(self, epoch: int) -> None:
        """Save checkpoint at the end of an epoch.

        Args:
            epoch: Current epoch (0-indexed).
        """
        checkpoint_dir = os.path.join(
            self.config["output_dir"], f"checkpoint-epoch-{epoch + 1}"
        )
        save_checkpoint(
            checkpoint_dir,
            self.model,
            self.accelerator,
            processor=self.processor,
            config=self.config,
        )
