#!/bin/bash
# CLIP ViT-B/32 — Text-to-Image with CoSENT + LoRA
source "$(dirname "$0")/_common.sh"
print_header "CLIP — t2i / cosent / lora"
resolve_data

python run.py examples/clip_cosent.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_clip_cosent/"
