#!/usr/bin/env bash
# Flickr30k before/after comparison (text-image retrieval)
# Usage:
#   ./compare_flickr30k_before_after.sh <BEFORE_MODEL> <AFTER_MODEL_PATH> [ENCODER_MODE] [BATCH_SIZE]
#
# Examples:
#   ./benchmark/compare_flickr30k_before_after.sh openai/clip-vit-base-patch32 ./experiments/output_clip/checkpoint-epoch-5
#   ./benchmark/compare_flickr30k_before_after.sh google/siglip-base-patch16-224 ./experiments/output_siglip/checkpoint-epoch-5
#   ./benchmark/compare_flickr30k_before_after.sh models/Qwen/Qwen3-VL-Embedding-2B ./experiments/output_qwen3_2b/checkpoint-epoch-1

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <BEFORE_MODEL> <AFTER_MODEL_PATH> [ENCODER_MODE] [BATCH_SIZE]"
    exit 1
fi

MODEL_BEFORE="$1"
MODEL_AFTER="$2"
ENCODER_MODE="${3:-}"
BATCH_SIZE="${4:-64}"
MRL_DIMS="${5:-}"

# Defaults that can be overridden by env vars
FLICKR_ROOT="${FLICKR_ROOT:-data/flickr30k}"
OUT_ROOT="${OUT_ROOT:-experiments/benchmark_output_flickr30k_compare}"

# Auto-detect encoder mode if not provided
if [[ -z "$ENCODER_MODE" ]]; then
    # Convert to lowercase for checking
    LOWER_BEFORE=$(echo "$MODEL_BEFORE" | tr '[:upper:]' '[:lower:]')
    LOWER_AFTER=$(echo "$MODEL_AFTER" | tr '[:upper:]' '[:lower:]')

    if [[ "$LOWER_BEFORE" == *"qwen"* ]] || [[ "$LOWER_AFTER" == *"qwen"* ]]; then
        ENCODER_MODE="qwen3_vl"
        # Auto-adjust batch size for Qwen if it's still default
        if [ "$BATCH_SIZE" == "64" ]; then
            echo "Note: Qwen detected, setting batch size to 64 (memory permitting)."
            BATCH_SIZE=64
        fi
    elif [[ "$LOWER_BEFORE" == *"siglip"* ]]; then
        ENCODER_MODE="siglip"
    else
        ENCODER_MODE="auto"
    fi
    echo "Auto-detected encoder mode: $ENCODER_MODE"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_NAME="flickr30k_report.json"

BOLD='\033[1m' DIM='\033[2m' RESET='\033[0m'
banner() { printf "\n${BOLD}%s${RESET}\n\n" "$*"; }

run_one() {
  local label="$1" model="$2" out_dir="$3"
  banner "${label}: ${model} (Mode: ${ENCODER_MODE})"
  mkdir -p "${out_dir}"
  
  # Construct command array to handle optional args safely
  local cmd=("python" "-u" "${SCRIPT_DIR}/run.py" "flickr30k")
  cmd+=("--model_path" "${model}")
  cmd+=("--flickr_root" "${FLICKR_ROOT}")
  cmd+=("--output_dir" "${out_dir}")
  cmd+=("--batch_size" "${BATCH_SIZE}")
  cmd+=("--encoder_mode" "${ENCODER_MODE}")
  
  if [ -n "${MRL_DIMS}" ]; then
      # Split space-separated string into individual arguments
      cmd+=("--mrl_dims" ${MRL_DIMS})
  fi

  "${cmd[@]}" | tee "${out_dir}/run.log"
}

run_one "BEFORE" "${MODEL_BEFORE}" "${OUT_ROOT}/before"
run_one "AFTER"  "${MODEL_AFTER}"  "${OUT_ROOT}/after"

banner "Comparison"
if [ -f "${SCRIPT_DIR}/compare_reports.py" ]; then
    python "${SCRIPT_DIR}/compare_reports.py" \
      "${OUT_ROOT}/before/${REPORT_NAME}" \
      "${OUT_ROOT}/after/${REPORT_NAME}"
else
    echo "Warning: compare_reports.py not found, skipping comparison summary."
    echo "Check results manually in:"
    echo "  Before: ${OUT_ROOT}/before/${REPORT_NAME}"
    echo "  After:  ${OUT_ROOT}/after/${REPORT_NAME}"
fi
