#!/bin/bash
# SigLIP — Text-to-Image with InfoNCE + LoRA
# Preset: siglip (batch=64, lr=5e-5). Pass --use_mrl to enable MRL [768,512,256]
source "$(dirname "$0")/_common.sh"
print_header "SigLIP — t2i / infonce / lora"
resolve_data

echo "DEBUG: IMAGE_ROOT is '$IMAGE_ROOT'"
echo "DEBUG: DATA_PATH is '$DATA_PATH'"

python run.py examples/siglip_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"
echo "Done: experiments/output_siglip/"
