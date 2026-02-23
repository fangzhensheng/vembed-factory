#!/bin/bash
# BGE-M3 — Text-to-Text retrieval (composed, symmetric)
# Preset: bge_m3 (composed, both encoders=BAAI/bge-base-en-v1.5)
source "$(dirname "$0")/_common.sh"
print_header "BGE-M3 composed — t2t / infonce / lora"
resolve_data true

python run.py examples/bge_t2t.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH"

echo "Done: experiments/output_bge_t2t/"
