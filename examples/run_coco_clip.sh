#!/bin/bash
# COCO + CLIP Training Script
# Prepares COCO dataset and trains CLIP model using the same pattern as run_clip.sh

source "$(dirname "$0")/_common.sh"
print_header "COCO + CLIP — t2i / infonce / lora"

# Prepare COCO dataset
# Default: auto-detect (prefer real data if available)
MODE=${1:-auto}

if [ "$MODE" = "hf" ]; then
    echo "Downloading COCO from HuggingFace..."
    python examples/prepare_data.py coco hf --split train,val --year 2017 --output data/coco
    DATA_PATH="data/coco/train.jsonl"
    VAL_DATA_PATH="data/coco/val.jsonl"
    IMAGE_ROOT="data/coco/images"

elif [ "$MODE" = "official" ]; then
    if [ -z "$2" ]; then
        echo "Error: official mode requires COCO dataset root path"
        echo "Usage: $0 official /path/to/coco/2017"
        exit 1
    fi
    DATASET_ROOT=$2
    echo "Using official COCO dataset: $DATASET_ROOT"
    python examples/prepare_data.py coco official "$DATASET_ROOT" --output data/coco
    DATA_PATH="data/coco/train.jsonl"
    VAL_DATA_PATH="data/coco/val.jsonl"
    IMAGE_ROOT="data/coco/images"

elif [ "$MODE" = "auto" ]; then
    # Auto-detect: use real COCO data if available
    if [ -f "data/coco/train.jsonl" ]; then
        echo "Auto-detected real COCO data"
        DATA_PATH="data/coco/train.jsonl"
        VAL_DATA_PATH="data/coco/val.jsonl"
        IMAGE_ROOT="data/coco/images"
    else
        echo "Error: no COCO data found at data/coco/train.jsonl"
        echo "  Options:"
        echo "    $0 hf         - Download from HuggingFace"
        echo "    $0 official   - Use local COCO dataset"
        exit 1
    fi

else
    echo "Unknown mode: $MODE"
    echo "  Available modes: auto (default), hf, official"
    exit 1
fi

echo "data: $DATA_PATH"
echo "val:  $VAL_DATA_PATH"
echo "imgs: $IMAGE_ROOT"
echo ""

# Train model using COCO-specific config (3 epochs)
python run.py examples/coco_clip_train.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --image_root "$IMAGE_ROOT"

echo "Done: experiments/output_clip/"
