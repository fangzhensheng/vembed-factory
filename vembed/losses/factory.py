from typing import Any

import torch.nn as nn

from . import functions  # noqa: F401 — trigger registration
from .functions.matryoshka import MatryoshkaLoss
from .registry import LossRegistry


class LossFactory:

    @staticmethod
    def create_distillation_loss(config: dict[str, Any]) -> nn.Module:
        try:
            loss_cls = LossRegistry.get("distillation")
        except ValueError:
            from .functions.distillation import DistillationLoss

            loss_cls = DistillationLoss
        return loss_cls(config)

    @staticmethod
    def create(config: dict[str, Any]) -> nn.Module:
        loss_type = config.get("loss_type", "infonce").lower()

        try:
            loss_cls = LossRegistry.get(loss_type)
        except ValueError as exc:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Available: {LossRegistry.list_losses()}"
            ) from exc

        base_loss = loss_cls(config)

        if not config.get("use_mrl", False):
            return base_loss

        return MatryoshkaLoss(
            base_loss,
            dims=config.get("mrl_dims", [768]),
            weights=config.get("mrl_weights"),
        )
