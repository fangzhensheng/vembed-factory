#!/bin/bash
# BGE-base — Text-to-Text with InfoNCE + LoRA
# Preset: bge (batch=64, lr=2e-5, lora=true)
source "$(dirname "$0")/_common.sh"
print_header "BGE — t2t / infonce / lora"
resolve_data "true"  # text_only=true

python run.py examples/bge_t2t.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH"

echo "Done: experiments/output_bge_t2t/"
