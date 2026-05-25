#!/usr/bin/env bash
# Continue the QWen MATH-500 best-of-n accuracy_verify sweep: vllm method at
# n = 16, 32, 64 (the ours legs and vllm n=2,4,8 are already done).
#
# Each run goes through scripts/test_time_compute.py (free decode, PRM scores
# every beam). Recipe is edited per-n (gemm_opt=false, chunk_size=none = vllm).
# Outputs default to data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/, then are moved
# into raw_data_accuracy_verify/ (jsonl+csv) and accuracy_verify/ (jsonl) to
# match the existing layout.
#
# Usage: bash scripts/run_vllm_accuracy_verify_n163264.sh

set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -n "${CONDA_SH:-}" ]]; then
  # Optional: set CONDA_SH to your local conda.sh before launching.
  source "$CONDA_SH"
  conda activate "${CONDA_ENV_NAME:-sal_new_vllm}"
fi
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

ts() { date '+%F %T'; }
LOG=logs/accuracy_verify_QWen_MATH500.log
RECIPE=recipes/best-of-n.yaml
DATA_DIR=data/Qwen/QWen2.5-1.5B-Instruct/MATH-500
RAW_DIR="$DATA_DIR/raw_data_accuracy_verify"
AV_DIR="$DATA_DIR/accuracy_verify"

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] CONTINUE: vllm legs n=16,32,64 (gemm_opt=false chunk_size=none)"
  echo "================================================================"
} >> "$LOG"

for n in 16 32 64; do
  # Configure recipe for this vllm run.
  sed -i "s/^n: .*/n: $n/" "$RECIPE"
  sed -i "s/^gemm_opt: .*/gemm_opt: false/" "$RECIPE"
  sed -i "s/^chunk_size: .*/chunk_size: none     # [dynamic, heuristic, none]/" "$RECIPE"
  sed -i "s/^disable_prm: .*/disable_prm: false/" "$RECIPE"

  echo "[$(ts)] VLLM_RUN_START  n=$n" >> "$LOG"
  python scripts/test_time_compute.py "$RECIPE"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] VLLM_RUN_FAIL   n=$n rc=$rc" >> "$LOG"
    exit $rc
  fi

  jsonl="$DATA_DIR/best_of_n_completions_vllm_n${n}.jsonl"
  csv="$DATA_DIR/best_of_n_request_timings_vllm_n${n}.csv"
  if [[ ! -f "$jsonl" ]]; then
    echo "[$(ts)] VLLM_RUN_FAIL   n=$n missing output $jsonl" >> "$LOG"
    exit 1
  fi
  rows=$(wc -l < "$jsonl")

  # Move outputs into the accuracy_verify layout.
  mv -f "$jsonl" "$RAW_DIR/"
  [[ -f "$csv" ]] && mv -f "$csv" "$RAW_DIR/"
  cp -f "$RAW_DIR/best_of_n_completions_vllm_n${n}.jsonl" "$AV_DIR/"

  echo "[$(ts)] VLLM_RUN_OK     n=$n rows=$rows -> raw_data_accuracy_verify/ + accuracy_verify/" >> "$LOG"
done

echo "[$(ts)] VLLM_LEGS_COMPLETE n=16,32,64 done" >> "$LOG"
