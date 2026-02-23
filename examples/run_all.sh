#!/bin/bash
# Master test runner — exercises every supported training configuration.
#
# Usage:
#   bash examples/run_all.sh              # run all tests
#   bash examples/run_all.sh models       # only model architecture tests
#   bash examples/run_all.sh losses       # only loss function tests
#   bash examples/run_all.sh modes        # only retrieval mode tests
#   bash examples/run_all.sh features     # only feature tests (flash, fsdp, wandb, etc.)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CATEGORY="${1:-all}"

PASSED=()
FAILED=()

run() {
    local script="$1"
    local label="$2"
    echo ""
    echo "RUNNING: $label ($script)"
    if bash "$SCRIPT_DIR/$script"; then
        PASSED+=("$label")
    else
        FAILED+=("$label")
        echo "FAILED: $label (continuing...)"
    fi
}

run_models() {
    run run_clip.sh              "CLIP ViT-B/32 (auto, t2i)"
    run run_siglip.sh            "SigLIP (auto, t2i)"
    run run_qwen3_2b.sh          "Qwen3-VL-2B (qwen3_vl, t2i, flash)"
    run run_qwen3_8b.sh          "Qwen3-VL-8B (qwen3_vl, t2i, flash)"
    run run_dinov2_bert.sh       "DINOv2+BERT (composed, t2i)"
    run run_bge_t2t.sh           "BGE-M3 (composed, t2t)"
    run run_bert_t2t.sh          "BERT (auto/custom, t2t)"
    run run_dinov2_i2i.sh        "DINOv2 (auto/custom, i2i)"
    run run_mae_i2i.sh           "MAE (auto/custom, i2i)"
}

run_losses() {
    run run_clip.sh              "InfoNCE (default)"
    run run_clip_triplet.sh      "Triplet (hardest-neg mining)"
    run run_clip_cosent.sh       "CoSENT (cosine sentence)"
    run run_qwen_colbert.sh      "ColBERT (late interaction, Qwen)"
    run run_dinov2_colbert.sh    "ColBERT (late interaction, DINOv2)"
    run run_distillation.sh      "Knowledge Distillation (KL)"
}

run_modes() {
    run run_clip.sh              "t2i (text-to-image)"
    run run_dinov2_i2i.sh        "i2i (image-to-image)"
    run run_bert_t2t.sh          "t2t (text-to-text)"
    run run_qwen_m2i.sh          "m2i (multimodal-to-image)"
    run run_qwen_m2t.sh          "m2t (multimodal-to-text)"
}

run_features() {
    run run_qwen3_8b_fsdp.sh     "FSDP multi-GPU (Qwen3-VL-8B)"
}

case "$CATEGORY" in
    models)   run_models ;;
    losses)   run_losses ;;
    modes)    run_modes ;;
    features) run_features ;;
    all)
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

echo ""
echo "RESULTS"
echo "Passed: ${#PASSED[@]}"
for t in "${PASSED[@]}"; do echo "  + $t"; done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED[@]}"
    for t in "${FAILED[@]}"; do echo "  - $t"; done
    exit 1
else
    echo "All tests passed."
fi
