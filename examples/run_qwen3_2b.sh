#!/bin/bash
# Qwen3-VL-Embedding-2B — Text-to-Image with LoRA + FlashAttention
# Config: examples/qwen3_2b_train.yaml
source "$(dirname "$0")/_common.sh"
print_header "Qwen3-VL-2B — t2i / infonce / lora / flash_attn"
resolve_data

# Use the YAML config file directly
# Override data paths from environment/script logic
python run.py examples/qwen3_2b_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen3_2b/"