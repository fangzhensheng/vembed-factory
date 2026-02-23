#!/bin/bash
# Qwen2-VL — ColBERT late-interaction loss (multi-vector retrieval)
# Requires colbert pooling strategy to return per-token embeddings
source "$(dirname "$0")/_common.sh"
print_header "Qwen2-VL — t2i / colbert / lora"
resolve_data

python run.py examples/qwen_colbert.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_qwen_colbert/"
