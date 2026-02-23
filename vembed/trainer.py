import logging

import torch

from vembed.cli import main as cli_main

logger = logging.getLogger(__name__)


class VEmbedFactoryTrainer:
    """High-level training API that wraps the CLI into a Python interface."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        mode: str = "auto",
        output_dir: str = "output",
        use_gpu: bool = True,
    ):
        self.model_name = model_name
        self.mode = self._detect_mode(model_name) if mode == "auto" else mode
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()

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
        config_override: str | None = None,
    ):
        """Start the training process by delegating to the CLI entrypoint."""
        cli_args: list[str] = [
            "--preset",
            self.mode,
            f"--model_name={self.model_name}",
            f"--data_path={data_path}",
            f"--output_dir={self.output_dir}",
            f"--epochs={epochs}",
            f"--batch_size={batch_size}",
            f"--lr={learning_rate}",
            f"--retrieval_mode={retrieval_mode}",
        ]

        if use_gradient_cache:
            cli_args.append("--use_gradient_cache")
        else:
            cli_args.append("--no_use_gradient_cache")

        if use_mrl:
            cli_args.append("--use_mrl")

        if report_to:
            cli_args.append(f"--report_to={report_to}")
        if attn_implementation:
            cli_args.append(f"--attn_implementation={attn_implementation}")
        if torch_dtype:
            cli_args.append(f"--torch_dtype={torch_dtype}")
        if gradient_checkpointing:
            cli_args.append("--gradient_checkpointing")

        overrides: list[str] = [
            f"model_name={self.model_name}",
            f"encoder_mode={encoder_mode}",
        ]
        if text_model_name:
            overrides.append(f"text_model_name={text_model_name}")
        if image_model_name:
            overrides.append(f"image_model_name={image_model_name}")
        if use_mrl and mrl_dims:
            overrides.append(f"mrl_dims={mrl_dims}")
        if val_data_path:
            overrides.append(f"val_data_path={val_data_path}")
        if save_steps > 0:
            overrides.append(f"save_steps={save_steps}")

        if config_override:
            overrides.append(config_override)

        if overrides:
            cli_args.append("--config_override")
            # Flatten overrides and handle splitting if needed
            for override in overrides:
                cli_args.extend(override.split())

        # Launch
        logger.info(f"Starting training for {self.model_name}...")
        logger.info(f"   Mode: {self.mode}, Retrieval: {retrieval_mode}")

        import sys

        sys.argv = [sys.argv[0]] + cli_args
        exit_code = cli_main(cli_args)

        if exit_code == 0:
            logger.info(f"Training finished successfully! Model saved to {self.output_dir}")
        else:
            logger.error(f"Training failed with exit code {exit_code}")

        return exit_code
