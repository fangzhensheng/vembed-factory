#!/bin/bash
# CLIP ViT-B/32 — Text-to-Image with InfoNCE + LoRA
# Preset: clip (batch=64, lr=5e-5, mrl=[512,256,128])
source "$(dirname "$0")/_common.sh"
print_header "CLIP — t2i / infonce / lora"
resolve_data

python run.py examples/clip_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_clip/"
