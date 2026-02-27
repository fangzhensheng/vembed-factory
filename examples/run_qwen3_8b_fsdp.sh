#!/bin/bash
# Qwen3-VL-8B — FSDP multi-GPU training (8B params sharded across GPUs)
# FSDP shards model parameters, gradients, and optimizer states across GPUs,
# enabling training of large VLMs that don't fit on a single GPU.
# Note: FSDP disables Gradient Cache automatically.
source "$(dirname "$0")/_common.sh"
print_header "Qwen3-VL-8B — t2i / infonce / fsdp / lora / flash_attn"
resolve_data

python run.py examples/qwen3_8b_fsdp.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen3_8b_fsdp/"
