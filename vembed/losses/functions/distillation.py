from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LossRegistry


@LossRegistry.register("distillation")
class DistillationLoss(nn.Module):
    """KL divergence (or MSE) on similarity matrices between teacher and student."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.temperature = config.get("distillation_temperature", 1.0)
        self.loss_type = config.get("distillation_loss_type", "kl").lower()

    def forward(
        self,
        student_query: torch.Tensor,
        student_positive: torch.Tensor,
        teacher_query: torch.Tensor,
        teacher_positive: torch.Tensor,
    ):
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
