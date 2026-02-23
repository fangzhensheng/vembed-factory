#!/bin/bash
# Knowledge Distillation: CLIP-Large (teacher) -> CLIP-Base (student)
# Student trained with MRL + LoRA, teacher frozen, KL-divergence on similarity matrices
source "$(dirname "$0")/_common.sh"
print_header "Distillation — CLIP-Large -> CLIP-Base / mrl / lora"
resolve_data

TEACHER="openai/clip-vit-large-patch14"
STUDENT="openai/clip-vit-base-patch32"

export CUDA_VISIBLE_DEVICES=0,1

python run.py examples/clip_distillation.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_distill/"
