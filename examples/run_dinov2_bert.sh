#!/bin/bash
# [EXPERIMENTAL] DINOv2 + BERT (Composed) — Text-to-Image
# Two independent encoders aligned via learnable projection heads.
# For production use-cases, prefer CLIP/SigLIP or Qwen3-VL.
# Preset: dinov2_bert (composed, batch=64, lr=5e-5, proj=512)
source "$(dirname "$0")/_common.sh"
print_header "DINOv2+BERT composed — t2i / infonce / lora / gc"
resolve_data

python run.py examples/dinov2_bert_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_dinov2_bert/"
