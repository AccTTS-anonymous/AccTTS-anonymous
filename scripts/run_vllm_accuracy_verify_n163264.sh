#!/usr/bin/env bash
# Multi-round best-of-n accuracy verification for MATH-500 on Llama-3.2-1B.
#
# Runs both methods:
#   vllm -> gemm_opt=false, chunk_size=none      (vLLM baseline)
#   ours -> gemm_opt=true,  chunk_size=dynamic   (AccTTS)
#
# PRM is enabled for every run (`disable_prm=false`) so `pred` is the
# PRM-ranked best-of-n answer. Outputs are renamed with the seed so repeated
# rounds do not overwrite each other:
#
#   data/meta-llama/Llama-3.2-1B-Instruct/MATH-500/accuracy_verify/
#     best_of_n_completions_vllm_seed0_n2.jsonl
#     best_of_n_completions_ours_seed0_n2.jsonl
#     ...
#
# Use scripts/accuracy_verify.ipynb afterward to compute averages/std error
# bars and generate the HuggingFace vs AccTTS plot.
#
# Usage:
#   bash scripts/run_vllm_accuracy_verify_n163264.sh
#
# Optional environment overrides:
#   SEEDS="0 1 2 3 4" BEAMS="2 4 8 16 32 64" OVERWRITE=1 bash ...
#   VLLM_GPU_MEMORY_UTILIZATION=0.5 PRM_BATCH_SIZE=1 bash ...

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-sal_new_vllm}"
fi

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN is not set. ${MODEL:-meta-llama/Llama-3.2-1B-Instruct} is gated; export HF_TOKEN or run huggingface-cli login before launching."
fi

ts() { date '+%F %T'; }

MODEL="meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME="HuggingFaceH4/MATH-500"
DATASET_SPLIT="test"
DATASET_SHORT="MATH-500"

RECIPE="recipes/best-of-n.yaml"
DATA_DIR="data/${MODEL}/${DATASET_SHORT}"
AV_DIR="${DATA_DIR}/accuracy_verify"
LOG="logs/accuracy_verify_Llama1B_MATH500_multi_round.log"

SEEDS=${SEEDS:-"0 1 2 3 4"}
BEAMS=${BEAMS:-"2 4 8 16 32 64"}
OVERWRITE=${OVERWRITE:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}
PRM_BATCH_SIZE=${PRM_BATCH_SIZE:-1}

mkdir -p "$AV_DIR" logs

run_one() {
  local method=$1 seed=$2 n=$3 gemm=$4 chunk=$5
  local final_jsonl="${AV_DIR}/best_of_n_completions_${method}_seed${seed}_n${n}.jsonl"
  local final_csv="${AV_DIR}/best_of_n_request_timings_${method}_seed${seed}_n${n}.csv"

  if [[ -f "$final_jsonl" && "$OVERWRITE" != "1" ]]; then
    echo "[$(ts)] SKIP existing method=${method} seed=${seed} n=${n}" | tee -a "$LOG"
    return 0
  fi

  echo "[$(ts)] RUN_START method=${method} seed=${seed} n=${n} gemm_opt=${gemm} chunk_size=${chunk}" | tee -a "$LOG"
  if ! python scripts/test_time_compute.py "$RECIPE" \
    --model_path="${MODEL}" \
    --dataset_name="${DATASET_NAME}" \
    --dataset_split="${DATASET_SPLIT}" \
    --n="${n}" \
    --seed="${seed}" \
    --gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
    --prm_batch_size="${PRM_BATCH_SIZE}" \
    --disable_prm=false \
    --gemm_opt="${gemm}" \
    --chunk_size="${chunk}" \
    >> "$LOG" 2>&1; then
    echo "[$(ts)] RUN_FAIL method=${method} seed=${seed} n=${n}; see ${LOG}" | tee -a "$LOG"
    echo "[$(ts)] Last log lines:" | tee -a "$LOG"
    tail -40 "$LOG"
    exit 1
  fi

  local src_jsonl="${DATA_DIR}/best_of_n_completions_${method}_n${n}.jsonl"
  local src_csv="${DATA_DIR}/best_of_n_request_timings_${method}_n${n}.csv"
  if [[ ! -f "$src_jsonl" ]]; then
    echo "[$(ts)] RUN_FAIL method=${method} seed=${seed} n=${n} missing ${src_jsonl}" | tee -a "$LOG"
    exit 1
  fi

  mv -f "$src_jsonl" "$final_jsonl"
  if [[ -f "$src_csv" ]]; then
    mv -f "$src_csv" "$final_csv"
  fi

  local rows
  rows=$(wc -l < "$final_jsonl")
  echo "[$(ts)] RUN_OK method=${method} seed=${seed} n=${n} rows=${rows} -> ${final_jsonl}" | tee -a "$LOG"
}

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] ACCURACY_VERIFY_START model=${MODEL} dataset=${DATASET_SHORT}"
  echo "[$(ts)] seeds=${SEEDS}"
  echo "[$(ts)] beams=${BEAMS}"
  echo "[$(ts)] vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} prm_batch_size=${PRM_BATCH_SIZE}"
  echo "================================================================"
} | tee -a "$LOG"

for seed in $SEEDS; do
  for n in $BEAMS; do
    run_one "vllm" "$seed" "$n" "false" "none"
    run_one "ours" "$seed" "$n" "true" "dynamic"
  done
done

echo "[$(ts)] ACCURACY_VERIFY_DONE" | tee -a "$LOG"
