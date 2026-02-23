#!/bin/bash
# CLIP ViT-B/32 — Text-to-Image with Triplet Loss
source "$(dirname "$0")/_common.sh"
print_header "CLIP — t2i / triplet / lora"
resolve_data

python run.py examples/clip_triplet.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_clip_triplet/"
