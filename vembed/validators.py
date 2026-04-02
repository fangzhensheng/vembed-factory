"""Parameter validation and error reporting for vembed-factory training.

Provides helpful error messages and validation for common configuration issues.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates training configuration and provides helpful error messages."""

    def __init__(self, config: dict[str, Any]):
        """Initialize validator with config dict."""
        self.config = config
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validations. Return True if valid, False if errors found."""
        self.errors = []
        self.warnings = []

        self._validate_model()
        self._validate_data()
        self._validate_loss_pooling()
        self._validate_distributed()
        self._validate_distillation()

        if self.errors:
            self._print_errors()
            return False

        if self.warnings:
            self._print_warnings()

        return True

    def _validate_model(self) -> None:
        """Validate model configuration."""
        model_name = self.config.get("model_name_or_path")
        if not model_name:
            self.errors.append("ERROR: model_name_or_path is required. " "See: vembed list-configs")

        # Validate composed mode
        encoder_mode = self.config.get("encoder_mode", "auto")
        if encoder_mode == "composed":
            if not self.config.get("text_model_name"):
                self.errors.append(
                    "ERROR: Composed mode requires --text_model_name "
                    "(e.g., --text_model_name BAAI/bge-base-en-v1.5)"
                )
            if not self.config.get("image_model_name"):
                self.errors.append(
                    "ERROR: Composed mode requires --image_model_name "
                    "(e.g., --image_model_name openai/clip-vit-base-patch32)"
                )

    def _validate_data(self) -> None:
        """Validate data configuration."""
        data_path = self.config.get("data_path")
        if not data_path:
            self.errors.append(
                "ERROR: data_path is required (path to training JSONL file). "
                "Use: vembed list-datasets  or  vembed show-dataset <name>"
            )
        elif not os.path.exists(data_path):
            self.errors.append(
                f"ERROR: Training data not found: {data_path}\n"
                f"   Run: vembed prepare-data msmarco_t2t"
            )

        # Check for image datasets that need image_root
        retrieval_mode = self.config.get("retrieval_mode", "t2i")
        if retrieval_mode in ("t2i", "i2i", "m2i"):
            image_root = self.config.get("image_root")
            if not image_root:
                self.warnings.append(
                    f"WARNING: retrieval_mode='{retrieval_mode}' usually needs --image_root"
                )
            elif not os.path.exists(image_root):
                self.warnings.append(f"WARNING: image_root directory not found: {image_root}")

    def _validate_loss_pooling(self) -> None:
        """Validate loss type and pooling method compatibility."""
        loss_type = self.config.get("loss_type", "infonce")
        pooling_method = self.config.get("pooling_method")

        if loss_type == "colbert":
            if pooling_method and pooling_method != "none":
                self.warnings.append(
                    f"WARNING: loss_type='colbert' works best with pooling_method='none' "
                    f"(not '{pooling_method}'). Auto-correcting."
                )
        else:
            if pooling_method == "none":
                self.warnings.append(
                    f"WARNING: loss_type='{loss_type}' requires global pooling "
                    f"(not 'none'). Auto-correcting to 'cls'."
                )

    def _validate_distributed(self) -> None:
        """Validate distributed training configuration."""
        use_fsdp = self.config.get("use_fsdp", False)
        use_gradient_cache = self.config.get("use_gradient_cache", True)

        if use_fsdp and use_gradient_cache:
            self.warnings.append(
                "INFO: Using both FSDP and Gradient Cache. This is fine for very large models."
            )

        if use_fsdp:
            num_gpus = self.config.get("num_gpus")
            if not num_gpus:
                self.warnings.append(
                    "WARNING: Using FSDP but num_gpus not specified. " "Set with --num_gpus <N>"
                )

    def _validate_distillation(self) -> None:
        """Validate knowledge distillation configuration."""
        teacher_model = self.config.get("teacher_model_name")
        if teacher_model:
            model_name = self.config.get("model_name_or_path")
            if model_name and model_name == teacher_model:
                self.warnings.append(
                    "WARNING: Student model same as teacher model. "
                    "Distillation will have no effect."
                )

    def _print_errors(self) -> None:
        """Print errors with formatting."""
        logger.error("=" * 70)
        logger.error("Configuration Validation Failed")
        logger.error("=" * 70)
        for error in self.errors:
            logger.error(error)
        logger.error("=" * 70)

    def _print_warnings(self) -> None:
        """Print warnings with formatting."""
        for warning in self.warnings:
            logger.warning(warning)


def validate_config(config: dict[str, Any]) -> bool:
    """Validate configuration and return success status."""
    validator = ConfigValidator(config)
    return validator.validate()
