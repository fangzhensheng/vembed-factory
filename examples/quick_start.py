"""Quick-start examples exercising the Python Trainer API.

Usage:
    python examples/quick_start.py          # all demos
    python examples/quick_start.py clip     # single demo by name
"""

import argparse
import sys
from pathlib import Path

from vembed import Trainer


def get_data_paths():
    """Resolve training and validation data paths with fallback."""
    flickr_train = Path("data/flickr30k/train.jsonl")
    dummy_train = Path("examples/dummy_data/train.jsonl")
    fallback_train = Path("data/dummy/train.jsonl")

    if flickr_train.exists():
        return str(flickr_train), str(Path("data/flickr30k/val.jsonl")), str(flickr_train.parent)

    if dummy_train.exists():
        return str(dummy_train), str(dummy_train), str(dummy_train.parent)

    # Fallback for very fresh clone without dummy data setup (should not happen if committed)
    return str(fallback_train), str(fallback_train), str(fallback_train.parent)


DATA, VAL, IMAGE_ROOT = get_data_paths()


def run_custom(args):
    """Run training with custom arguments."""
    trainer = Trainer(args.model_name, output_dir=args.output_dir)
    trainer.train(
        data_path=DATA,
        epochs=args.epochs,
        batch_size=args.batch_size,
        retrieval_mode=args.retrieval_mode,
        encoder_mode=args.encoder_mode,
        config_override=f"pooling_method={args.pooling} image_root={IMAGE_ROOT}",
    )


def clip_basic():
    """Minimal CLIP training."""
    trainer = Trainer("openai/clip-vit-base-patch32", output_dir="experiments/output_quick_clip")
    trainer.train(data_path=DATA, epochs=1, config_override=f"image_root={IMAGE_ROOT}")


def siglip_lora():
    """SigLIP with LoRA fine-tuning."""
    trainer = Trainer(
        "google/siglip-base-patch16-224", output_dir="experiments/output_quick_siglip"
    )
    trainer.train(
        data_path=DATA, epochs=1, use_lora=True, config_override=f"image_root={IMAGE_ROOT}"
    )


def clip_flash_attn():
    """CLIP with FlashAttention-2 + bfloat16."""
    trainer = Trainer(
        "openai/clip-vit-base-patch32", output_dir="experiments/output_quick_clip_flash"
    )
    trainer.train(
        data_path=DATA,
        epochs=1,
        use_lora=True,
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        config_override=f"image_root={IMAGE_ROOT}",
    )


def clip_mrl():
    """CLIP with Matryoshka Representation Learning."""
    trainer = Trainer(
        "openai/clip-vit-base-patch32", output_dir="experiments/output_quick_clip_mrl"
    )
    trainer.train(
        data_path=DATA,
        epochs=1,
        use_lora=True,
        use_mrl=True,
        config_override=f"image_root={IMAGE_ROOT}",
    )


def clip_distill():
    """Knowledge distillation: CLIP-Large -> CLIP-Base."""
    trainer = Trainer("openai/clip-vit-base-patch32", output_dir="experiments/output_quick_distill")
    trainer.train(
        data_path=DATA,
        val_data_path=VAL,
        epochs=1,
        use_lora=True,
        use_mrl=True,
        config_override=(
            'teacher_model_name="openai/clip-vit-large-patch14" '
            "distillation_alpha=0.5 "
            "distillation_temperature=2.0 "
            f"image_root={IMAGE_ROOT}"
        ),
    )


def qwen3_vl():
    """Qwen3-VL-2B with FlashAttention."""
    trainer = Trainer(
        "models/Qwen/Qwen3-VL-Embedding-2B", output_dir="experiments/output_quick_qwen3"
    )
    trainer.train(
        data_path=DATA,
        epochs=1,
        use_lora=True,
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        config_override=f"image_root={IMAGE_ROOT}",
    )


DEMOS = {
    "clip": clip_basic,
    "siglip": siglip_lora,
    "flash": clip_flash_attn,
    "mrl": clip_mrl,
    "distill": clip_distill,
    "qwen3": qwen3_vl,
}


def main():
    """CLI entrypoint."""
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Handle custom arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default="output")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--retrieval_mode", type=str, default="t2i")
        parser.add_argument("--encoder_mode", type=str, default="auto")
        parser.add_argument("--pooling", type=str, default="mean")
        args = parser.parse_args()
        run_custom(args)
        return

    requested = sys.argv[1] if len(sys.argv) > 1 else None

    if requested:
        if requested not in DEMOS:
            print(f"Unknown demo '{requested}'. Available: {', '.join(DEMOS)}")
            sys.exit(1)
        DEMOS[requested]()
        return

    for name, fn in DEMOS.items():
        print(f"\n{'='*60}\n  Demo: {name}\n{'='*60}")
        fn()


if __name__ == "__main__":
    main()
