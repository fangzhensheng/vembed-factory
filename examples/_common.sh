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
    # Priority: Flickr30k full > Flickr30k in examples > Dummy data
    local text_only="${1:-false}"

    if [ -f "data/flickr30k/train.jsonl" ]; then
        DATA_PATH="data/flickr30k/train.jsonl"
        IMAGE_ROOT="data/flickr30k"
        [ -f "data/flickr30k/val.jsonl" ] && VAL_DATA_PATH="data/flickr30k/val.jsonl"
    elif [ -f "examples/data/flickr30k/train.jsonl" ]; then
        DATA_PATH="examples/data/flickr30k/train.jsonl"
        IMAGE_ROOT="examples/data/flickr30k"
        [ -f "examples/data/flickr30k/val.jsonl" ] && VAL_DATA_PATH="examples/data/flickr30k/val.jsonl"
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
        echo "Error: no training data found. Run: python examples/prepare_data.py"
        exit 1
    fi

    echo "data: $DATA_PATH"
    [ -n "$VAL_DATA_PATH" ] && echo "val:  $VAL_DATA_PATH"
    [ "$text_only" = "false" ] && echo "imgs: $IMAGE_ROOT"
}

print_header() {
    echo "$1"
}
