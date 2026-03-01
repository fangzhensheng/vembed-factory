#!/usr/bin/env bash
# SOP before/after comparison (Image-to-Image retrieval)
# Unified script for DINOv2, MAE, DINOv3, and ColBERT evaluation on Stanford Online Products.
#
# Usage:
#   ./compare_sop_before_after.sh <BEFORE_MODEL> <AFTER_MODEL> [SIMILARITY_MODE] [BATCH_SIZE] [TOPK]
#
# Modes (SIMILARITY_MODE):
#   cosine   : (Default) Standard dense retrieval (DINOv2, MAE, CLIP, DINOv3).
#   colbert  : Before runs 'cosine' (baseline), After runs 'colbert' (MaxSim).
#
# Examples:
#   # 1. DINOv2 (Cosine)
#   ./benchmark/compare_sop_before_after.sh facebook/dinov2-small ./experiments/output_sop_dinov2_i2i/checkpoint-epoch-20 cosine
#
#   # 2. MAE (Cosine)
#   ./benchmark/compare_sop_before_after.sh facebook/vit-mae-base ./experiments/output_sop_mae_i2i/checkpoint-epoch-37 cosine
#
#   # 3. DINOv3 (Cosine)
#   ./benchmark/compare_sop_before_after.sh models/dinov3-vitb16-pretrain-lvd1689m ./experiments/output_sop_dinov3_i2i/checkpoint-epoch-16 cosine
#
#   # 4. ColBERT (Before=Cosine, After=MaxSim)
#   ./benchmark/compare_sop_before_after.sh facebook/dinov2-base ./experiments/output_sop_dinov2_colbert/checkpoint-epoch-3 colbert

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <BEFORE_MODEL> <AFTER_MODEL> [SIMILARITY_MODE] [BATCH_SIZE] [TOPK]"
    exit 1
fi

MODEL_BEFORE="$1"
MODEL_AFTER="$2"
SIMILARITY_MODE="${3:-cosine}"
BATCH_SIZE="${4:-256}"
TOPK="${5:-100}"
MRL_DIMS="${6:-}"

# Defaults
SOP_ROOT="${SOP_ROOT:-data/stanford_online_products}"
OUT_ROOT="${OUT_ROOT:-experiments/benchmark_output_sop_compare}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BOLD='\033[1m' DIM='\033[2m' RESET='\033[0m'
banner() { printf "\n${BOLD}%s${RESET}\n\n" "$*"; }

# Helper to run evaluation
run_eval() {
    local label="$1"
    local model="$2"
    local out_dir="$3"
    local sim_type="$4"
    
    mkdir -p "${out_dir}"
    banner "${label}: ${model} (${sim_type})"

    # Construct command array to handle optional args safely
    local cmd=("python" "-u" "${SCRIPT_DIR}/run.py" "sop")
    cmd+=("--model_path" "${model}")
    cmd+=("--sop_root" "${SOP_ROOT}")
    cmd+=("--output_dir" "${out_dir}")
    cmd+=("--batch_size" "${BATCH_SIZE}")
    cmd+=("--topk" "${TOPK}")

    if [ "$sim_type" == "colbert" ]; then
        cmd+=("--similarity" "colbert" "--pooling" "none")
    else
        cmd+=("--similarity" "cosine")
    fi
    
    if [ -n "${MRL_DIMS}" ]; then
        # Split space-separated string into individual arguments
        cmd+=("--mrl_dims" ${MRL_DIMS})
    fi
    
    "${cmd[@]}" | tee "${out_dir}/run.log"
}

# Logic for different modes
if [ "$SIMILARITY_MODE" == "colbert" ]; then
    # ColBERT Mode: Before is dense baseline, After is ColBERT
    run_eval "BEFORE (Baseline)" "${MODEL_BEFORE}" "${OUT_ROOT}/before" "cosine"
    run_eval "AFTER  (ColBERT)"  "${MODEL_AFTER}"  "${OUT_ROOT}/after"  "colbert"
    
    REPORT_BEFORE="sop_standard_report.json"
    REPORT_AFTER="sop_colbert_report.json"
else
    # Standard Mode: Both are dense
    run_eval "BEFORE" "${MODEL_BEFORE}" "${OUT_ROOT}/before" "cosine"
    run_eval "AFTER"  "${MODEL_AFTER}"  "${OUT_ROOT}/after"  "cosine"
    
    REPORT_BEFORE="sop_standard_report.json"
    REPORT_AFTER="sop_standard_report.json"
fi

banner "Comparison"
if [ -f "${SCRIPT_DIR}/compare_reports.py" ]; then
    python "${SCRIPT_DIR}/compare_reports.py" \
      "${OUT_ROOT}/before/${REPORT_BEFORE}" \
      "${OUT_ROOT}/after/${REPORT_AFTER}"
else
    echo "Warning: compare_reports.py not found."
fi
