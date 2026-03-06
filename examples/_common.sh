#!/bin/bash
# Shared setup sourced by all training scripts.
# Provides: PROJECT_ROOT, DATA_PATH, IMAGE_ROOT, VAL_DATA_PATH, resolve_data()

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

DATA_PATH=""
IMAGE_ROOT=""
VAL_DATA_PATH=""

resolve_data() {
    # Priority: For t2t: msmarco > dummy; For t2i: flickr30k > dummy
    local text_only="${1:-false}"

    # Check for t2t datasets first if text_only=true
    if [ "$text_only" = "true" ]; then
        if [ -f "data/msmarco/train.jsonl" ]; then
            DATA_PATH="data/msmarco/train.jsonl"
            [ -f "data/msmarco/val.jsonl" ] && VAL_DATA_PATH="data/msmarco/val.jsonl"
            echo "data: $DATA_PATH"
            [ -n "$VAL_DATA_PATH" ] && echo "val:  $VAL_DATA_PATH"
            return
        elif [ -f "examples/data/msmarco/train.jsonl" ]; then
            DATA_PATH="examples/data/msmarco/train.jsonl"
            [ -f "examples/data/msmarco/val.jsonl" ] && VAL_DATA_PATH="examples/data/msmarco/val.jsonl"
            echo "data: $DATA_PATH"
            [ -n "$VAL_DATA_PATH" ] && echo "val:  $VAL_DATA_PATH"
            return
        fi
    fi

    # Image-text datasets (t2i or i2i)
    if [ -f "data/flickr30k/train.jsonl" ]; then
        DATA_PATH="data/flickr30k/train.jsonl"
        IMAGE_ROOT="data/flickr30k"
        [ -f "data/flickr30k/val.jsonl" ] && VAL_DATA_PATH="data/flickr30k/val.jsonl"
    elif [ -f "examples/data/flickr30k/train.jsonl" ]; then
        DATA_PATH="examples/data/flickr30k/train.jsonl"
        IMAGE_ROOT="examples/data/flickr30k"
        [ -f "examples/data/flickr30k/val.jsonl" ] && VAL_DATA_PATH="examples/data/flickr30k/val.jsonl"
    elif [ -f "data/stanford_online_products/train.jsonl" ]; then
        DATA_PATH="data/stanford_online_products/train.jsonl"
        IMAGE_ROOT="data/stanford_online_products"
        [ -f "data/stanford_online_products/val.jsonl" ] && VAL_DATA_PATH="data/stanford_online_products/val.jsonl"
    elif [ -f "examples/dummy_data/train.jsonl" ]; then
        DATA_PATH="examples/dummy_data/train.jsonl"
        IMAGE_ROOT="examples/dummy_data"
    elif [ -f "data/dummy/train.jsonl" ]; then
        DATA_PATH="data/dummy/train.jsonl"
        IMAGE_ROOT="data/dummy"
    elif [ -f "examples/data/dummy/train.jsonl" ]; then
        DATA_PATH="examples/data/dummy/train.jsonl"
        IMAGE_ROOT="examples/data/dummy"
    else
        echo "Error: no training data found. Run: python examples/prepare_data.py msmarco_t2t"
        exit 1
    fi

    echo "data: $DATA_PATH"
    [ -n "$VAL_DATA_PATH" ] && echo "val:  $VAL_DATA_PATH"
    [ "$text_only" = "false" ] && echo "imgs: $IMAGE_ROOT"
}

print_header() {
    echo "$1"
}
