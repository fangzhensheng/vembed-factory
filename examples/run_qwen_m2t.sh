#!/bin/bash
# Qwen2-VL — Multimodal-to-Text retrieval (m2t)
# Query = text + image (multimodal), Positive = text
source "$(dirname "$0")/_common.sh"
print_header "Qwen2-VL — m2t / infonce / lora"
resolve_data

python run.py examples/qwen_m2t.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen_m2t/"
