#!/usr/bin/env bash
# Sweep Qwen2.5-7B-Instruct on a fixed 100-sample MATH-500 subset, then full AIME.
#
# For each (algorithm, dataset, n), this script:
#   1. collects a fresh vLLM trace with task_trace and keeps its vLLM timing,
#   2. replays that exact trace with ours settings.
#
# Usage:
#   bash scripts/max_tokens_4090_run.sh [all|math500|aime|bon_math500|bon_aime|bs_math500|bs_aime]

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SEED=${SEED:-0}
MATH500_NUM_SAMPLES=${MATH500_NUM_SAMPLES:-100}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_TOKENS=${MAX_TOKENS:-4096}
NS=(${NS:-2 4 8 16 32})

LOG_DIR=${LOG_DIR:-logs/max_tokens_4090_run}
MASTER_LOG=${MASTER_LOG:-logs/max_tokens_4090_run.log}
mkdir -p "$LOG_DIR" logs

ts() { date '+%F %T'; }

log() {
  echo "[$(ts)] $*" | tee -a "$MASTER_LOG"
}

set_system_prompt_preset() {
  local preset=$1
  "$PY" - "$preset" <<'PY'
import pathlib
import re
import sys

preset = sys.argv[1]
assert preset in ("AIME", "MATH500"), preset
p = pathlib.Path("src/sal/config.py")
s = p.read_text()
s = re.sub(
    r'^_SYSTEM_PROMPT_PRESET = .*$',
    f'_SYSTEM_PROMPT_PRESET = "{preset}"',
    s,
    flags=re.M,
)
p.write_text(s)
PY
}

set_recipe() {
  local recipe=$1 approach=$2 ds=$3 n=$4 gemm=$5 chunk=$6
  local bw=${7:-}
  "$PY" - "$recipe" "$approach" "$ds" "$n" "$gemm" "$chunk" "$bw" \
      "$MODEL" "$SEED" "$MATH500_NUM_SAMPLES" "$GPU_MEMORY_UTILIZATION" "$MAX_TOKENS" <<'PY'
import pathlib
import re
import sys

(
    recipe,
    approach,
    ds,
    n,
    gemm,
    chunk,
    bw,
    model,
    seed,
    math500_num_samples,
    gpu_memory_utilization,
    max_tokens,
) = sys.argv[1:13]

if ds == "math500":
    dataset_name = "HuggingFaceH4/MATH-500"
    dataset_split = "test"
    num_samples = math500_num_samples
elif ds == "aime":
    dataset_name = "AI-MO/aimo-validation-aime"
    dataset_split = "train"
    num_samples = None
else:
    raise ValueError(ds)

p = pathlib.Path(recipe)
s = p.read_text()

def set_key(text, key, value):
    line = f"{key}: {value}"
    pattern = rf"^\s*#?\s*{re.escape(key)}: .*$"
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, line, text, count=1, flags=re.M)
    return text.rstrip() + "\n" + line + "\n"

def remove_key(text, key):
    return re.sub(rf"^\s*#?\s*{re.escape(key)}: .*\n?", "", text, flags=re.M)

s = set_key(s, "dataset_name", dataset_name)
s = set_key(s, "dataset_split", dataset_split)
s = set_key(s, "model_path", model)
s = set_key(s, "gpu_memory_utilization", gpu_memory_utilization)
s = set_key(s, "approach", approach)
s = set_key(s, "n", n)
s = set_key(s, "search_batch_size", "1")
s = set_key(s, "seed", seed)
s = set_key(s, "max_tokens", max_tokens)
s = set_key(s, "gemm_opt", gemm)
s = set_key(s, "chunk_size", chunk)
s = set_key(s, "disable_prm", "true")
s = set_key(s, "push_to_hub", "false")
s = remove_key(s, "dataset_start")
s = remove_key(s, "dataset_end")

if num_samples is None:
    s = remove_key(s, "num_samples")
else:
    s = set_key(s, "num_samples", num_samples)

if approach == "beam_search":
    s = set_key(s, "beam_width", bw)

p.write_text(s)
PY
}

run_cmd() {
  local label=$1
  local logfile=$2
  shift 2

  log "${label}_START log=${logfile}"
  if "$@" > "$logfile" 2>&1; then
    log "${label}_OK"
  else
    local rc=$?
    log "${label}_FAIL rc=$rc log=${logfile}"
    tail -40 "$logfile" | tee -a "$MASTER_LOG"
    exit $rc
  fi
}

dataset_meta() {
  local ds=$1
  if [[ "$ds" == "aime" ]]; then
    DATASET_PRESET="AIME"
    DATASET_TAG="aime"
  else
    DATASET_PRESET="MATH500"
    DATASET_TAG="math500_100"
  fi
}

run_bon_one() {
  local ds=$1 n=$2
  dataset_meta "$ds"

  set_recipe "recipes/best-of-n.yaml" "best_of_n" "$ds" "$n" "false" "none"
  run_cmd "BON_TRACE_VLLM ds=${ds} n=${n}" \
    "$LOG_DIR/bon_${DATASET_TAG}_trace_vllm_n${n}.log" \
    "$PY" scripts/best_of_n_task_trace.py recipes/best-of-n.yaml

  set_recipe "recipes/best-of-n.yaml" "best_of_n" "$ds" "$n" "true" "dynamic"
  run_cmd "BON_REPLAY_OURS ds=${ds} n=${n}" \
    "$LOG_DIR/bon_${DATASET_TAG}_replay_ours_n${n}.log" \
    "$PY" scripts/test_time_compute.py recipes/best-of-n.yaml
}

run_bs_one() {
  local ds=$1 n=$2 bw
  dataset_meta "$ds"
  if [[ $n -eq 2 ]]; then bw=2; else bw=4; fi

  set_recipe "recipes/beam-search.yaml" "beam_search" "$ds" "$n" "false" "none" "$bw"
  run_cmd "BS_TRACE_VLLM ds=${ds} n=${n} bw=${bw}" \
    "$LOG_DIR/bs_${DATASET_TAG}_trace_vllm_n${n}.log" \
    "$PY" scripts/beam_search_task_trace.py recipes/beam-search.yaml

  set_recipe "recipes/beam-search.yaml" "beam_search" "$ds" "$n" "true" "dynamic" "$bw"
  run_cmd "BS_REPLAY_OURS ds=${ds} n=${n} bw=${bw}" \
    "$LOG_DIR/bs_${DATASET_TAG}_replay_ours_n${n}.log" \
    "$PY" scripts/beam_search_beam.py recipes/beam-search.yaml
}

run_bon_phase() {
  local ds=$1
  dataset_meta "$ds"

  set_system_prompt_preset "$DATASET_PRESET"
  log "BON_PHASE_START ds=${ds} preset=${DATASET_PRESET} seed=${SEED}"

  for n in "${NS[@]}"; do
    run_bon_one "$ds" "$n"
  done

  log "BON_PHASE_DONE ds=${ds}"
}

run_bs_phase() {
  local ds=$1
  dataset_meta "$ds"

  set_system_prompt_preset "$DATASET_PRESET"
  log "BS_PHASE_START ds=${ds} preset=${DATASET_PRESET} seed=${SEED}"

  for n in "${NS[@]}"; do
    run_bs_one "$ds" "$n"
  done

  log "BS_PHASE_DONE ds=${ds}"
}

run_dataset_phase() {
  local ds=$1
  dataset_meta "$ds"

  set_system_prompt_preset "$DATASET_PRESET"
  log "DATASET_PHASE_START ds=${ds} preset=${DATASET_PRESET} seed=${SEED}"

  for n in "${NS[@]}"; do
    log "DATASET_BEAM_START ds=${ds} n=${n}"
    run_bon_one "$ds" "$n"
    run_bs_one "$ds" "$n"
    log "DATASET_BEAM_DONE ds=${ds} n=${n}"
  done

  log "DATASET_PHASE_DONE ds=${ds}"
}

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] MAX_TOKENS_4090_RUN_START"
  echo "[$(ts)] model=${MODEL}"
  echo "[$(ts)] seed=${SEED} math500_num_samples=${MATH500_NUM_SAMPLES}"
  echo "[$(ts)] ns=${NS[*]}"
  echo "[$(ts)] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} max_tokens=${MAX_TOKENS}"
  echo "================================================================"
} | tee -a "$MASTER_LOG"

PHASE=${1:-all}
case "$PHASE" in
  bon_math500)
    run_bon_phase math500
    ;;
  bon_aime)
    run_bon_phase aime
    ;;
  bs_math500)
    run_bs_phase math500
    ;;
  bs_aime)
    run_bs_phase aime
    ;;
  math500)
    run_dataset_phase math500
    ;;
  aime)
    run_dataset_phase aime
    ;;
  all)
    run_dataset_phase math500
    run_dataset_phase aime
    ;;
  *)
    echo "Usage: $0 [all|math500|aime|bon_math500|bon_aime|bs_math500|bs_aime]" >&2
    exit 2
    ;;
esac

log "MAX_TOKENS_4090_RUN_DONE phase=${PHASE}"
