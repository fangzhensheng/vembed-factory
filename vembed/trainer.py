import logging

import torch

from vembed.config import load_base_config, merge_configs
from vembed.entrypoints.train import train_entrypoint
from vembed.training.config import prepare_output_dir

logger = logging.getLogger(__name__)


class VEmbedTrainer:
    """Flexible training API for custom components."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        mode: str = "auto",
        output_dir: str = "output",
        use_gpu: bool = True,
        loss_type: str = "infonce",
        collator_type: str | None = None,
    ):
        """Initialize trainer.

        Args:
            model_name: Model name or path.
            mode: Encoder mode ('auto', 'clip', 'qwen', etc.).
            output_dir: Directory to save outputs.
            use_gpu: Whether to use GPU.
            loss_type: Registered loss function name.
            collator_type: Registered collator name.
        """
        self.model_name = model_name
        self.mode = self._detect_mode(model_name) if mode == "auto" else mode
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_type = loss_type
        self.collator_type = collator_type

    @staticmethod
    def _detect_mode(model_name: str) -> str:
        lower = model_name.lower()
        for keyword, mode in [("clip", "clip"), ("siglip", "siglip"), ("qwen", "qwen")]:
            if keyword in lower:
                return mode
        return "custom"

    def train(
        self,
        data_path: str,
        val_data_path: str | None = None,
        epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 5e-5,
        use_gradient_cache: bool = True,
        use_mrl: bool = False,
        mrl_dims: list[int] | None = None,
        retrieval_mode: str = "t2i",
        encoder_mode: str = "auto",
        text_model_name: str | None = None,
        image_model_name: str | None = None,
        save_steps: int = 0,
        use_lora: bool = True,
        report_to: str | None = None,
        attn_implementation: str | None = None,
        torch_dtype: str | None = None,
        gradient_checkpointing: bool = False,
        image_root: str | None = None,
        **kwargs,
    ) -> dict:
        """Train model with direct entrypoint (no CLI layer).

        Returns:
            Dictionary with training results including output_dir and model_path.
        """
        logger.info(f"Starting training for {self.model_name}...")
        logger.info(f"   Mode: {self.mode}, Retrieval: {retrieval_mode}")

        # Load defaults and merge with user config
        defaults = load_base_config()

        user_config = {
            "model_name": self.model_name,
            "data_path": data_path,
            "output_dir": self.output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "retrieval_mode": retrieval_mode,
            "use_gradient_cache": use_gradient_cache,
            "use_mrl": use_mrl,
            "use_lora": use_lora,
            "gradient_checkpointing": gradient_checkpointing,
        }

        if self.loss_type and self.loss_type != "infonce":
            user_config["loss_type"] = self.loss_type

        resolved_encoder_mode = self.collator_type if self.collator_type else encoder_mode
        if resolved_encoder_mode and resolved_encoder_mode != "auto":
            user_config["encoder_mode"] = resolved_encoder_mode

        if val_data_path:
            user_config["val_data_path"] = val_data_path
        if save_steps > 0:
            user_config["save_steps"] = save_steps

        if text_model_name:
            user_config["text_model_name"] = text_model_name
        if image_model_name:
            user_config["image_model_name"] = image_model_name

        if report_to:
            user_config["report_to"] = report_to
        if attn_implementation:
            user_config["attn_implementation"] = attn_implementation
        if torch_dtype:
            user_config["torch_dtype"] = torch_dtype
        if image_root:
            user_config["image_root"] = image_root

        if use_mrl and mrl_dims:
            user_config["mrl_dims"] = mrl_dims

        user_config.update(kwargs)

        config = merge_configs(defaults, {}, {}, user_config)
        prepare_output_dir(config)

        try:
            result = train_entrypoint(config, accelerator=None)
            logger.info(f"Training finished successfully! Model saved to {result['output_dir']}")
            return result
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
