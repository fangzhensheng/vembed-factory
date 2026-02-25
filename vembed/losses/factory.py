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

        # Initialize gather based on loss type and config
        # InfoNCE defaults to gather enabled, others default to disabled
        # Can be overridden via config "enable_gather" key
        from .functions.base import BaseLoss
        if isinstance(base_loss, BaseLoss):
            base_loss.set_gather(base_loss.enable_gather_default)

        if not config.get("use_mrl", False):
            return base_loss

        mrl_loss = MatryoshkaLoss(
            base_loss,
            dims=config.get("mrl_dims", [768]),
            weights=config.get("mrl_weights"),
        )

        # Propagate gather setting to the MRL wrapper
        if isinstance(base_loss, BaseLoss):
            mrl_loss.set_gather(base_loss.enable_gather_default)

        return mrl_loss
