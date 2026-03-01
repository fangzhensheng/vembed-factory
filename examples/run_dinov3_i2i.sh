#!/bin/bash
# DINOv3 — Image-to-Image retrieval on SOP dataset
# Auto model (DINOv3) with i2i retrieval mode + InfoNCE (Label-Aware)

set -e

cd "$(dirname "$0")/.." || exit 1

# # Set single GPU mode
# export CUDA_VISIBLE_DEVICES=0

# Source common functions (optional, if exists)
if [ -f "$(dirname "$0")/_common.sh" ]; then
    source "$(dirname "$0")/_common.sh"
fi

echo "DINOv3 on SOP (i2i)"

# SOP Dataset Paths (relative to project root)
SOP_ROOT="data/stanford_online_products"
DATA_PATH="${SOP_ROOT}/train.jsonl"
VAL_DATA_PATH="${SOP_ROOT}/val.jsonl"
IMAGE_ROOT="${SOP_ROOT}"

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run: python examples/prepare_data.py sop_i2i"
    exit 1
fi

# Configuration
CONFIG_FILE="examples/dinov3_i2i.yaml"
OUTPUT_DIR="experiments/output_sop_dinov3_i2i"

# Launch training
python run.py "$CONFIG_FILE" \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo ""
echo "Done. Results saved to $OUTPUT_DIR"
