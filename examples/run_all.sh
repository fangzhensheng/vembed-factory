#!/bin/bash
# Master test runner — exercises every supported training configuration.
# Each test runs for 1 epoch to keep runtime short.
# Configured for single GPU with 24GB memory limit.
#
# Usage:
#   bash examples/run_all.sh              # run all tests (1 epoch each, single GPU)
#   bash examples/run_all.sh models       # only model architecture tests
#   bash examples/run_all.sh losses       # only loss function tests
#   bash examples/run_all.sh modes        # only retrieval mode tests
#   bash examples/run_all.sh features     # only feature tests (flash, fsdp, wandb, etc.)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CATEGORY="${1:-all}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source common functions for data resolution
source "$SCRIPT_DIR/_common.sh"

PASSED=()
FAILED=()

# Resolve data paths before running tests
resolve_data

# ========== Environment Configuration ==========
# Single GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Memory limits (24GB consumer GPU) — managed via debug_gpu_memory parameter in CLI
# PYTORCH_CUDA_ALLOC_CONF reduces fragmentation but doesn't hard-limit memory
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Optimization flags
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_DETERMINISTIC=1

# ========== Test Runner Function ==========
# Run trainer with 1 epoch override and memory constraints
# Usage: run_test <yaml_config> <label> [extra_args...]
run_test() {
    local yaml="$1"
    local label="$2"
    shift 2
    local extra_args="$@"

    echo ""
    echo "=========================================="
    echo "RUNNING: $label"
    echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (Single GPU)"
    echo "Memory: 24GB limit (debug_gpu_memory=24)"
    echo "=========================================="

    cd "$PROJECT_ROOT"
    if python run.py "examples/$yaml" \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --image_root "$IMAGE_ROOT" \
        --config_override epochs=1 debug_gpu_memory=24 \
        $extra_args; then
        PASSED+=("$label")
        echo "✓ PASSED: $label"
    else
        FAILED+=("$label")
        echo "✗ FAILED: $label (continuing...)"
    fi
}

# ========== Test Suites ==========

run_models() {
    run_test clip_train.yaml                          "CLIP ViT-B/32 (auto, t2i)"
    run_test siglip_train.yaml                        "SigLIP (auto, t2i)"
    run_test qwen3_vl_embedding_2b_train.yaml         "Qwen3-VL-2B (qwen3_vl, t2i, flash)"
    run_test qwen3_vl_embedding_8b_train.yaml         "Qwen3-VL-8B (qwen3_vl, t2i, flash)"
    run_test dinov2_bert_train.yaml                   "DINOv2+BERT (composed, t2i)"
    run_test bge_t2t.yaml                             "BGE-M3 (composed, t2t)"
    run_test bert_t2t.yaml                            "BERT (auto/custom, t2t)"
    run_test dinov2_i2i.yaml                          "DINOv2 (auto/custom, i2i)"
    run_test dinov3_i2i.yaml                          "DINOv3 (auto/custom, i2i)"
    run_test mae_i2i.yaml                             "MAE (auto/custom, i2i)"
}

run_losses() {
    run_test clip_train.yaml                          "InfoNCE (default)"
    run_test clip_triplet.yaml                        "Triplet (hardest-neg mining)"
    run_test clip_cosent.yaml                         "CoSENT (cosine sentence)"
    run_test qwen_colbert.yaml                        "ColBERT (late interaction, Qwen)"
    run_test dinov2_colbert.yaml                      "ColBERT (late interaction, DINOv2)"
    run_test clip_distillation.yaml                   "Knowledge Distillation (KL)"
}

run_modes() {
    run_test clip_train.yaml                          "t2i (text-to-image)"
    run_test dinov2_i2i.yaml                          "i2i (image-to-image / dinov2)"
    run_test dinov3_i2i.yaml                          "i2i (image-to-image / dinov3)"
    run_test bert_t2t.yaml                            "t2t (text-to-text)"
    run_test qwen_m2i.yaml                            "m2i (multimodal-to-image)"
    run_test qwen_m2t.yaml                            "m2t (multimodal-to-text)"
}

run_features() {
    run_test qwen3_vl_embedding_8b_fsdp.yaml          "FSDP multi-GPU (Qwen3-VL-8B)"
}

# ========== Main Execution ==========

case "$CATEGORY" in
    models)   run_models ;;
    losses)   run_losses ;;
    modes)    run_modes ;;
    features) run_features ;;
    all)
        echo "=========================================="
        echo "Running ALL test suites (1 epoch each)"
        echo "GPU: Single GPU (CUDA:0)"
        echo "Memory: 24GB limit"
        echo "Data: $DATA_PATH"
        echo "Images: $IMAGE_ROOT"
        echo "=========================================="
        run_models
        run_losses
        run_modes
        run_features
        ;;
    *)
        echo "Usage: $0 {all|models|losses|modes|features}"
        exit 1
        ;;
esac

# ========== Results Summary ==========

echo ""
echo "=========================================="
echo "TEST RESULTS SUMMARY"
echo "=========================================="
echo "Total Passed: ${#PASSED[@]}"
for t in "${PASSED[@]}"; do echo "  ✓ $t"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Total Failed: ${#FAILED[@]}"
    for t in "${FAILED[@]}"; do echo "  ✗ $t"; done
    echo "=========================================="
    exit 1
else
    echo ""
    echo "🎉 All tests passed!"
    echo "=========================================="
fi
