"""Validation and evaluation utilities."""

from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

from vembed.evaluation.metrics import compute_recall_metrics
from vembed.training.data_utils import maybe_first, unpack_positive_batch, unpack_query_batch

logger = get_logger(__name__)


class Evaluator:
    """Handles model evaluation and validation loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Any,
        accelerator: Accelerator,
        retrieval_mode: str = "t2i",
        log_with: str | None = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            criterion: Loss function.
            accelerator: Accelerate instance for distributed evaluation.
            retrieval_mode: Retrieval mode (e.g., 't2i', 'i2i', 'm2i').
            log_with: Optional tracker name for logging metrics.
        """
        self.model = model
        self.criterion = criterion
        self.accelerator = accelerator
        self.retrieval_mode = retrieval_mode
        self.log_with = log_with

    def evaluate(
        self,
        val_dataloader: Any,
        global_step: int = 0,
    ) -> float:
        """Run validation loop and compute metrics.

        Args:
            val_dataloader: Validation dataloader.
            global_step: Current training step for logging.

        Returns:
            Average validation loss.
        """
        self.model.eval()

        # Free training-related GPU memory (gradients, optimizer states) before eval
        torch.cuda.empty_cache()

        total_loss, num_batches = 0.0, 0
        all_q_embs, all_p_embs, all_q_labels, all_p_labels = [], [], [], []
        self.accelerator.print("\nRunning validation...")

        with torch.no_grad():
            for batch in tqdm(val_dataloader, disable=not self.accelerator.is_local_main_process):
                q_batch = unpack_query_batch(batch, self.retrieval_mode)
                p_batch = unpack_positive_batch(batch, self.retrieval_mode)

                q_embs = maybe_first(self.model(**q_batch))
                p_embs = maybe_first(self.model(**p_batch))

                loss_kwargs = {}
                has_labels = "labels" in batch
                if has_labels:
                    loss_kwargs["labels"] = batch["labels"]
                    labels = batch["labels"]

                total_loss += self.criterion(q_embs, p_embs, None, **loss_kwargs).item()
                num_batches += 1

                # Delete batch objects early to prevent GPU memory accumulation
                del q_batch, p_batch, batch
                torch.cuda.empty_cache()

                # Move embeddings to CPU immediately to avoid GPU memory buildup over iterations
                all_q_embs.append(self.accelerator.gather_for_metrics(q_embs).cpu())
                del q_embs
                torch.cuda.empty_cache()

                all_p_embs.append(self.accelerator.gather_for_metrics(p_embs).cpu())
                del p_embs
                torch.cuda.empty_cache()

                if has_labels:
                    all_q_labels.append(self.accelerator.gather_for_metrics(labels).cpu())
                    torch.cuda.empty_cache()
                    all_p_labels.append(self.accelerator.gather_for_metrics(labels).cpu())
                    del labels
                    torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        self.accelerator.print(f"Validation loss: {avg_loss:.4f}")

        # Compute recall metrics if labels are available
        _ = self._compute_and_log_metrics(
            all_q_embs, all_p_embs, all_q_labels, all_p_labels, global_step
        )

        self.model.train()
        return avg_loss

    def _compute_and_log_metrics(
        self,
        all_q_embs: list[torch.Tensor],
        all_p_embs: list[torch.Tensor],
        all_q_labels: list[torch.Tensor],
        all_p_labels: list[torch.Tensor],
        global_step: int,
    ) -> dict[str, float]:
        """Compute recall metrics and log to tracker.

        Args:
            all_q_embs: Query embeddings from all batches.
            all_p_embs: Positive embeddings from all batches.
            all_q_labels: Query labels from all batches.
            all_p_labels: Positive labels from all batches.
            global_step: Current training step.

        Returns:
            Dictionary of recall metrics.
        """
        recall_metrics = {}
        if all_q_labels and all_q_embs:
            recall_metrics = compute_recall_metrics(
                all_q_embs,
                all_p_embs,
                all_q_labels,
                all_p_labels if all_p_labels else None,
                k_list=[1, 10, 100],
                exclude_diagonal=False,
            )
            self.accelerator.print("Validation recalls:")
            for metric_name, metric_val in recall_metrics.items():
                self.accelerator.print(f"  {metric_name}: {metric_val:.4f}")
            self.accelerator.print()

        # Log metrics to tracker
        if self.log_with is not None:
            log_dict = {"val/loss": 0.0}  # Loss will be added by trainer
            log_dict.update(recall_metrics)
            self.accelerator.log(log_dict, step=global_step)

        return recall_metrics
