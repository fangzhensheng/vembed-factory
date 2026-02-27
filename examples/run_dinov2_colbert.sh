#!/bin/bash
# DINOv2 ColBERT — Image-to-Image retrieval on SOP dataset
# Custom model with i2i retrieval mode + ColBERT Loss

# Source common functions (optional, if exists)
if [ -f "$(dirname "$0")/_common.sh" ]; then
    source "$(dirname "$0")/_common.sh"
fi

echo "DINOv2 ColBERT on SOP (i2i)"

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

# Training Configuration
# Use dinov2-base to match the InfoNCE baseline (dinov2-small also works but weaker)
MODEL_NAME="facebook/dinov2-base"
OUTPUT_DIR="experiments/output_sop_dinov2_colbert"

echo "Model: $MODEL_NAME | projection_dim=128, pooling=none, topk_tokens=32"

python run.py examples/dinov2_colbert.yaml \
  --data_path "$DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --image_root "$IMAGE_ROOT"

echo "Training completed. Results saved to $OUTPUT_DIR"
