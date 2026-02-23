#!/bin/bash
# Benchmark entrypoint (dataset-oriented).
#
# Usage:
#   bash benchmark/run_benchmark.sh <DATASET> [DATASET_ARGS...]
#
# Datasets:
#   - jsonl: generic jsonl retrieval benchmark (t2i/i2i/i2t/t2t)
#       bash benchmark/run_benchmark.sh jsonl <MODEL_PATH> <DATA_PATH> <IMAGE_ROOT> [OUTPUT_DIR] [RETRIEVAL_MODE]
#   - sop: Stanford Online Products standard i2i Recall@K on Ebay_test.txt
#       bash benchmark/run_benchmark.sh sop <MODEL_PATH> [SOP_ROOT] [OUTPUT_DIR] [TOPK]
set -e

DATASET="${1:-}"
shift || true

if [ -z "$DATASET" ]; then
    echo "Usage: $0 <DATASET> [DATASET_ARGS...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$DATASET" in
    jsonl)
        MODEL_PATH="${1:-}"
        DATA_PATH="${2:-}"
        IMAGE_ROOT="${3:-}"
        OUTPUT_DIR="${4:-experiments/benchmark_output}"
        RETRIEVAL_MODE="${5:-t2i}"

        if [ -z "$MODEL_PATH" ] || [ -z "$DATA_PATH" ] || [ -z "$IMAGE_ROOT" ]; then
            echo "Usage: $0 jsonl <MODEL_PATH> <DATA_PATH> <IMAGE_ROOT> [OUTPUT_DIR] [RETRIEVAL_MODE]"
            exit 1
        fi

        echo "Dataset: $DATASET"
        echo "Model:   $MODEL_PATH"
        echo "Data:    $DATA_PATH"
        echo "Images:  $IMAGE_ROOT"
        echo "Mode:    $RETRIEVAL_MODE"
        echo "Output:  $OUTPUT_DIR"

        echo ""
        echo "Step 1/2: Generating embeddings..."
        accelerate launch "$SCRIPT_DIR/generate_embeddings.py" \
            --model_path "$MODEL_PATH" \
            --dataset jsonl \
            --data_path "$DATA_PATH" \
            --image_root "$IMAGE_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 64 \
            --retrieval_mode "$RETRIEVAL_MODE"

        echo ""
        echo "Step 2/2: Evaluating with dense cosine (from .npy)..."
        python "$SCRIPT_DIR/run.py" npy \
            --query_path "$OUTPUT_DIR/test_query_embeddings.npy" \
            --doc_path "$OUTPUT_DIR/test_doc_embeddings.npy" \
            --output_dir "$OUTPUT_DIR"

        echo ""
        echo "Done. Results in $OUTPUT_DIR/"
        ;;
    sop)
        MODEL_PATH="${1:-}"
        SOP_ROOT="${2:-data/stanford_online_products}"
        OUTPUT_DIR="${3:-experiments/benchmark_output_sop}"
        TOPK="${4:-100}"

        if [ -z "$MODEL_PATH" ]; then
            echo "Usage: $0 sop <MODEL_PATH> [SOP_ROOT] [OUTPUT_DIR] [TOPK]"
            exit 1
        fi

        echo "Dataset: $DATASET"
        echo "Model:   $MODEL_PATH"
        echo "SOP:     $SOP_ROOT"
        echo "TopK:    $TOPK"
        echo "Output:  $OUTPUT_DIR"

        echo ""
        python "$SCRIPT_DIR/run.py" sop \
            --model_path "$MODEL_PATH" \
            --sop_root "$SOP_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --topk "$TOPK"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported datasets: jsonl, sop"
        exit 1
        ;;
esac
