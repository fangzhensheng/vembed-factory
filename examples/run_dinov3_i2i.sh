#!/bin/bash
# DINOv3 — Image-to-Image retrieval on SOP dataset
# Auto model (DINOv3) with i2i retrieval mode + InfoNCE (Label-Aware)

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
    echo "Please run: python3 examples/prepare_sop_data.py"
    exit 1
fi

# Configuration
CONFIG_FILE="examples/dinov3_i2i.yaml"
OUTPUT_DIR="experiments/output_sop_dinov3_i2i"

# Launch training
# Using accelerate launch for distributed training support if needed
# But run.py is the entrypoint. The example script used `python run.py`.
# We should stick to that unless we want multi-gpu.
# The user's example used `python run.py`.

python run.py "$CONFIG_FILE" \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done. Results saved to $OUTPUT_DIR"
