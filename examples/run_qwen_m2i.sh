#!/bin/bash
# Qwen2-VL — Multimodal-to-Image retrieval (m2i)
# Query = text + image (multimodal), Positive = image
source "$(dirname "$0")/_common.sh"
print_header "Qwen2-VL — m2i / infonce / lora"
resolve_data

python run.py examples/qwen_m2i.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen_m2i/"
