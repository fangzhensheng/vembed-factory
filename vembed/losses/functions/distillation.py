from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry
from .base import BaseLoss


@LossRegistry.register("distillation")
class DistillationLoss(BaseLoss):
    """KL divergence (or MSE) on similarity matrices between teacher and student.

    Gather is disabled by default for Distillation loss.
    """

    enable_gather_default: bool = False

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get("distillation_temperature", 1.0)
        self.loss_type = config.get("distillation_loss_type", "kl").lower()
        self._enable_gather = config.get("enable_gather", self.enable_gather_default)

    def _forward(
        self,
        student_query: torch.Tensor,
        student_positive: torch.Tensor,
        teacher_query: torch.Tensor = None,
        teacher_positive: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass for distillation loss.

        Note: Distillation has a different signature than standard contrastive losses.
        When used through the standard interface, teacher embeddings should be passed
        as keyword arguments.
        """
        # Handle both direct call and standard interface
        if teacher_query is None:
            teacher_query = kwargs.get("teacher_query")
        if teacher_positive is None:
            teacher_positive = kwargs.get("teacher_positive")

        assert teacher_query is not None and teacher_positive is not None, (
            "Distillation loss requires teacher_query and teacher_positive embeddings"
        )

        student_query = F.normalize(student_query, p=2, dim=1)
        student_positive = F.normalize(student_positive, p=2, dim=1)
        teacher_query = F.normalize(teacher_query, p=2, dim=1)
        teacher_positive = F.normalize(teacher_positive, p=2, dim=1)

        student_sim = student_query @ student_positive.T
        teacher_sim = teacher_query @ teacher_positive.T

        if self.loss_type == "mse":
            return F.mse_loss(student_sim, teacher_sim)

        if self.loss_type == "kl":
            student_log_probs = F.log_softmax(student_sim / self.temperature, dim=1)
            with torch.no_grad():
                teacher_probs = F.softmax(teacher_sim / self.temperature, dim=1)
            # T^2 scaling is standard for distillation (Hinton et al.)
            return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
                self.temperature**2
            )

        raise ValueError(f"Unknown distillation loss type: {self.loss_type}")

    def forward(
        self,
        student_query: torch.Tensor,
        student_positive: torch.Tensor,
        negative_emb: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass interface compatible with standard contrastive loss interface.

        For distillation, negative_emb is not used. Teacher embeddings should be passed
        via kwargs: teacher_query and teacher_positive.
        """
        if self.enable_gather:
            # Gather all embeddings to compute loss on global batch
            student_query = self._gather_tensor(student_query, axis=0)
            student_positive = self._gather_tensor(student_positive, axis=0)

            if "teacher_query" in kwargs:
                kwargs["teacher_query"] = self._gather_tensor(kwargs["teacher_query"], axis=0)
            if "teacher_positive" in kwargs:
                kwargs["teacher_positive"] = self._gather_tensor(kwargs["teacher_positive"], axis=0)

        return self._forward(student_query, student_positive, **kwargs)
