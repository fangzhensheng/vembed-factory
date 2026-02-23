from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_name: str | None = field(default=None, metadata={"help": "Alias for model_name_or_path."})
    model_type: str | None = field(
        default="custom", metadata={"help": "Model type (e.g., 'clip', 'qwen', 'custom')"}
    )
    encoder_mode: str | None = field(
        default="auto", metadata={"help": "Encoder mode (e.g., 'auto', 'clip_like', 'composed')"}
    )
    text_model_name: str | None = field(
        default=None, metadata={"help": "Name or path of the text model (for composed models)"}
    )
    image_model_name: str | None = field(
        default=None, metadata={"help": "Name or path of the image model (for composed models)"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning"})
    lora_r: int = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA Alpha value."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA Dropout value."})
    lora_target_modules: list[str] | None = field(
        default=None, metadata={"help": "List of module names to apply LoRA to."}
    )
    teacher_model_name: str | None = field(
        default=None, metadata={"help": "Teacher model name for distillation."}
    )
    projection_dim: int | None = field(
        default=None, metadata={"help": "Dimension to project embeddings to"}
    )
    torch_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype."
        },
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={
            "help": "Attention implementation to use (e.g., 'flash_attention_2', 'sdpa', 'eager')"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str | None = field(
        default=None, metadata={"help": "Path to the training data file (JSONL)."}
    )
    val_data_path: str | None = field(
        default=None, metadata={"help": "Path to the validation data file (JSONL)."}
    )
    image_root: str | None = field(
        default=None,
        metadata={"help": "Root directory for images (if paths in data are relative)."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to training loop.
    """

    output_dir: str | None = field(
        default="output_run",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    lr: float | None = field(
        default=None, metadata={"help": "Alias for learning_rate (for backward compatibility)."}
    )
    epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    loss_type: str = field(
        default="infonce",
        metadata={"help": "The loss function to use (infonce, mrl, triplet, etc.)."},
    )
    retrieval_mode: str = field(
        default="t2i", metadata={"help": "Retrieval mode (t2i, i2i, t2t, etc.)."}
    )
    use_gradient_cache: bool = field(
        default=True, metadata={"help": "Whether to use Gradient Cache for memory efficiency."}
    )
    gradient_cache_chunk_size: int = field(
        default=4, metadata={"help": "Chunk size for Gradient Cache."}
    )
    use_mrl: bool = field(
        default=False, metadata={"help": "Whether to use Matryoshka Representation Learning."}
    )
    mrl_dims: list[int] | None = field(
        default=None, metadata={"help": "Dimensions for MRL training."}
    )
    mrl_weights: list[float] | None = field(
        default=None, metadata={"help": "Weights for MRL loss at each dimension."}
    )
    report_to: str | None = field(
        default=None,
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    save_steps: int = field(default=0, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    num_gpus: int | None = field(
        default=None, metadata={"help": "Number of GPUs to use (for launch command generation)."}
    )
    dry_run: bool = field(
        default=False, metadata={"help": "If True, generate config but do not launch training."}
    )
    use_fsdp: bool = field(
        default=False, metadata={"help": "Whether to use FSDP (Fully Sharded Data Parallel)."}
    )

    # Missing args from error log
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory"}
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "The checkpoint save strategy to use."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    max_image_size: int = field(
        default=224, metadata={"help": "The maximum image size for processing."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use."})
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    fp16: bool = field(
        default=False, metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit."}
    )
    topk_tokens: int | None = field(
        default=None, metadata={"help": "Top-K tokens for ColBERT-like models."}
    )
    temperature: float = field(default=0.05, metadata={"help": "Temperature for contrastive loss."})
    distillation_alpha: float = field(
        default=0.5, metadata={"help": "Alpha for distillation loss."}
    )
    distillation_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for distillation."}
    )
    distillation_loss_type: str = field(
        default="kl", metadata={"help": "Distillation loss type (kl, mse, etc.)."}
    )
