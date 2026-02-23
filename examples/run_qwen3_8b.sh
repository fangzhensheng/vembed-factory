#!/bin/bash
# Qwen3-VL-Embedding-8B — Text-to-Image with LoRA + FlashAttention
# Preset: qwen3_8b (encoder_mode=qwen3_vl, batch=1, lr=1e-5, gc_chunk=1, mrl=[3584])
source "$(dirname "$0")/_common.sh"
print_header "Qwen3-VL-8B — t2i / infonce / lora / flash_attn"
resolve_data

python run.py examples/qwen3_8b_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen3_8b/"
