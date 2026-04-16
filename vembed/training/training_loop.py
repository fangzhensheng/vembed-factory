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
    """Orchestrates the training loop with support for distributed training, gradient caching, and distillation.

    Gradient Accumulation:
        Uses accelerator.accumulate(model) for proper distributed training support.
        This handles DDP synchronization, loss scaling, and boundary cases automatically.
        Simply wrap the training step with: `with self.accelerator.accumulate(self.model):`
    """

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
        training_state: dict[str, Any] | None = None,
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
            training_state: Optional training state dict from resumed checkpoint
                with keys: global_step, epoch, best_metric, patience_counter.
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

        # Gradient accumulation and validation config
        self.grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
        self.eval_steps = int(config.get("eval_steps", 0))
        self.early_stopping_patience = int(config.get("early_stopping_patience", -1))
        self.eval_metric = config.get("eval_metric", "val/loss")
        self.eval_metric_better = config.get("eval_metric_better", "min")

        # Early stopping state tracking
        self.best_metric = float("inf") if self.eval_metric_better == "min" else float("-inf")
        self.patience_counter = 0

        # Resume training from checkpoint if provided
        self.global_step = 0
        self.current_epoch = 0
        if training_state is not None:
            self.global_step = training_state.get("global_step", 0)
            self.current_epoch = training_state.get("epoch", 0)
            self.best_metric = training_state.get("best_metric", self.best_metric)
            self.patience_counter = training_state.get("patience_counter", 0)
            logger.info(
                f"Resumed from checkpoint: global_step={self.global_step}, "
                f"epoch={self.current_epoch}, best_metric={self.best_metric}"
            )

    def train(self) -> None:
        """Run the complete training loop."""
        self.model.train()
        global_step = self.global_step
        steps_per_epoch = len(self.dataloader)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.accelerator.print(f"Epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(
                tqdm(self.dataloader, disable=not self.accelerator.is_local_main_process)
            ):
                try:
                    with self.accelerator.accumulate(self.model):
                        loss_val = self._train_step(batch)
                        global_step += 1

                        if global_step % self.logging_steps == 0:
                            self._log_step(global_step, loss_val, epoch, step, steps_per_epoch)

                        if self.save_steps > 0 and global_step % self.save_steps == 0:
                            self._save_checkpoint(global_step)

                        # Mid-epoch validation
                        if (
                            self.eval_steps > 0
                            and global_step % self.eval_steps == 0
                            and self.val_dataloader
                        ):
                            metrics_dict = self.evaluator.evaluate(self.val_dataloader, global_step)
                            if self.accelerator.log_with is not None:
                                self.accelerator.log(
                                    {"val/" + k: v for k, v in metrics_dict.items()},
                                    step=global_step,
                                )
                            val_metric = self._extract_eval_metric(metrics_dict)
                            if self._check_early_stopping(val_metric):
                                self.accelerator.print(
                                    f"Early stopping triggered at step {global_step}"
                                )
                                return
                except RuntimeError as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        batch_size = 0
                        if isinstance(batch, dict) and "input_ids" in batch:
                            batch_size = batch["input_ids"].shape[0]
                        elif isinstance(batch, dict) and "pixel_values" in batch:
                            batch_size = batch["pixel_values"].shape[0]

                        self.accelerator.print("\n" + "=" * 70)
                        self.accelerator.print("OUT OF MEMORY ERROR")
                        self.accelerator.print("=" * 70)
                        self.accelerator.print(
                            f"Step {global_step}, Epoch {epoch + 1}/{self.num_epochs}"
                        )
                        self.accelerator.print(f"Batch size: {batch_size}")
                        self.accelerator.print("\nRecommended Solutions:")
                        self.accelerator.print(f"  1. Reduce batch_size (current: {batch_size})")
                        self.accelerator.print("  2. Enable gradient_accumulation_steps")
                        self.accelerator.print("  3. Enable gradient_checkpointing: true")
                        self.accelerator.print("  4. Use smaller model")
                        self.accelerator.print("  5. Reduce max_seq_length or image_size")
                        self.accelerator.print("=" * 70 + "\n")
                    raise

            # Save checkpoint after each epoch
            self._save_checkpoint_epoch(epoch)

            # Epoch-end validation
            if self.val_dataloader and self.eval_steps == 0:
                metrics_dict = self.evaluator.evaluate(self.val_dataloader, global_step)
                if self.accelerator.log_with is not None:
                    self.accelerator.log(
                        {"val/" + k: v for k, v in metrics_dict.items()}, step=global_step
                    )
                val_metric = self._extract_eval_metric(metrics_dict)
                if self._check_early_stopping(val_metric):
                    self.accelerator.print(f"Early stopping triggered at epoch {epoch + 1}")
                    return

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

        Optimization: Use accelerator.no_sync() to skip gradient synchronization
        in non-final accumulation steps, reducing communication overhead.

        Args:
            batch: Input batch.

        Returns:
            Loss value.
        """
        # Gradient cache automatically handles batch unpacking and loss computation
        # with gradient sync optimization based on distributed training setup
        loss_val = self.grad_cache.step(self.model, batch)

        if self.accelerator.sync_gradients and self.max_grad_norm > 0:
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
            q_embs, p_embs, n_embs = self._forward_concatenated(q_inputs, p_inputs, n_inputs)
        else:
            use_no_sync = bool(self.accelerator and self.accelerator.num_processes > 1)
            
            if use_no_sync and hasattr(self.model, "no_sync"):
                with self.model.no_sync():
                    q_embs = maybe_first(self.model(**q_inputs))
                    if n_inputs:
                        p_embs = maybe_first(self.model(**p_inputs))
                
                # The last forward pass MUST NOT be in no_sync() to trigger DDP sync
                if n_inputs:
                    n_embs = maybe_first(self.model(**n_inputs))
                else:
                    p_embs = maybe_first(self.model(**p_inputs))
                    n_embs = None
            else:
                q_embs = maybe_first(self.model(**q_inputs))
                p_embs = maybe_first(self.model(**p_inputs))
                n_embs = maybe_first(self.model(**n_inputs)) if n_inputs else None

        loss_kwargs = {}
        if "labels" in batch:
            loss_kwargs["labels"] = batch["labels"]

        loss = self.criterion(q_embs, p_embs, n_embs, **loss_kwargs)

        if self.teacher_model is not None and self.distillation_loss_fn is not None:
            loss = self._apply_distillation(batch, q_embs, p_embs, loss)

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients and self.max_grad_norm > 0:
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
        if (
            self.processor
            and hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer.pad_token_id is not None
        ):
            return self.processor.tokenizer.pad_token_id

        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "pad_token_id")
            and self.model.config.pad_token_id is not None
        ):
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
            t_q = maybe_first(self.teacher_model(**unpack_query_batch(batch, self.retrieval_mode)))
            t_p = maybe_first(
                self.teacher_model(**unpack_positive_batch(batch, self.retrieval_mode))
            )

        distill_loss = self.distillation_loss_fn(q_embs, p_embs, t_q, t_p)
        loss = (
            self.distillation_alpha * student_loss + (1.0 - self.distillation_alpha) * distill_loss
        )
        return loss

    def _extract_eval_metric(self, metrics_dict: dict[str, float]) -> float:
        """Extract the configured eval_metric from full metrics dictionary.

        Supports custom metrics specified in eval_metric config:
        - "loss": uses metrics_dict["loss"]
        - "recall@1": uses metrics_dict["recall@1"]
        - "recall@5": uses metrics_dict["recall@5"]
        - etc.

        Args:
            metrics_dict: Full metrics dictionary from evaluator (e.g., {"loss": 0.5, "recall@1": 0.7})

        Returns:
            The specific metric value for early stopping.

        Raises:
            KeyError: If configured eval_metric not found in metrics_dict.
        """
        metric_name = self.eval_metric.replace("val/", "")
        if metric_name not in metrics_dict:
            available = list(metrics_dict.keys())
            self.accelerator.print(
                f"WARNING: eval_metric '{metric_name}' not found in validation metrics. "
                f"Available: {available}. Using 'loss' as fallback."
            )
            return metrics_dict.get("loss", float("inf"))
        return metrics_dict[metric_name]

    def _check_early_stopping(self, metric_value: float) -> bool:
        """Check if early stopping should be triggered based on configured metric.

        CRITICAL: Uses eval_metric_better config to determine improvement direction.
        - "min": Lower is better (default, for loss)
        - "max": Higher is better (for recall@k, accuracy, etc.)

        Args:
            metric_value: The metric value to evaluate (from _extract_eval_metric).

        Returns:
            True if early stopping should be triggered, False otherwise.
        """
        if self.early_stopping_patience < 0:
            return False

        is_better = (
            metric_value < self.best_metric
            if self.eval_metric_better == "min"
            else metric_value > self.best_metric
        )

        if is_better:
            self.best_metric = metric_value
            self.patience_counter = 0
            self.accelerator.print(
                f"Metric improved to {metric_value:.4f}. Best: {self.best_metric:.4f}"
            )
            return False

        self.patience_counter += 1
        self.accelerator.print(
            f"Metric did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}"
        )
        return self.patience_counter >= self.early_stopping_patience

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
        self.accelerator.print(f"  step {global_step} | loss={loss_val:.4f} | lr={current_lr:.2e}")
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
        checkpoint_dir = os.path.join(self.config["output_dir"], f"checkpoint-step-{global_step}")

        training_state = {
            "global_step": global_step,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
        }

        save_checkpoint(
            checkpoint_dir,
            self.model,
            self.accelerator,
            processor=self.processor,
            config=self.config,
            training_state=training_state,
        )

    def _save_checkpoint_epoch(self, epoch: int) -> None:
        """Save checkpoint at the end of an epoch.

        Args:
            epoch: Current epoch (0-indexed).
        """
        checkpoint_dir = os.path.join(self.config["output_dir"], f"checkpoint-epoch-{epoch + 1}")

        training_state = {
            "global_step": self.global_step,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
        }

        save_checkpoint(
            checkpoint_dir,
            self.model,
            self.accelerator,
            processor=self.processor,
            config=self.config,
            training_state=training_state,
        )
