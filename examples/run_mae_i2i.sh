#!/bin/bash
# MAE — Image-to-Image retrieval on SOP dataset
# Custom model with i2i retrieval mode + InfoNCE (Label-Aware)

# Source common functions (optional, if exists)
if [ -f "$(dirname "$0")/_common.sh" ]; then
    source "$(dirname "$0")/_common.sh"
fi

echo "MAE on SOP (i2i)"

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

# Training Configuration
MODEL_NAME="facebook/vit-mae-base"
OUTPUT_DIR="experiments/output_sop_mae_i2i"

# Launch training
python run.py examples/mae_i2i.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done. Results saved to $OUTPUT_DIR"
