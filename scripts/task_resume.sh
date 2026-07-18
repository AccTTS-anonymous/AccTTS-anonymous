#!/usr/bin/env bash
# Resume AIME n=128 only: best-of-n trace/replay, then beam-search trace/replay.

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
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_TOKENS=${MAX_TOKENS:-2048}
N=${N:-128}

LOG_DIR=${LOG_DIR:-logs/task_resume}
MASTER_LOG=${MASTER_LOG:-logs/task_resume.log}
mkdir -p "$LOG_DIR" logs

ts() { date '+%F %T'; }

log() {
  echo "[$(ts)] $*" | tee -a "$MASTER_LOG"
}

set_system_prompt_preset() {
  "$PY" - <<'PY'
import pathlib
import re

p = pathlib.Path("src/sal/config.py")
s = p.read_text()
s = re.sub(r'^_SYSTEM_PROMPT_PRESET = .*$', '_SYSTEM_PROMPT_PRESET = "AIME"', s, flags=re.M)
p.write_text(s)
PY
}

set_recipe() {
  local recipe=$1 approach=$2 n=$3 gemm=$4 chunk=$5
  local bw=${6:-}
  "$PY" - "$recipe" "$approach" "$n" "$gemm" "$chunk" "$bw" \
      "$MODEL" "$SEED" "$GPU_MEMORY_UTILIZATION" "$MAX_TOKENS" <<'PY'
import pathlib
import re
import sys

recipe, approach, n, gemm, chunk, bw, model, seed, gpu_memory_utilization, max_tokens = sys.argv[1:11]
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

s = set_key(s, "dataset_name", "AI-MO/aimo-validation-aime")
s = set_key(s, "dataset_split", "train")
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
s = remove_key(s, "num_samples")
s = remove_key(s, "dataset_start")
s = remove_key(s, "dataset_end")

if approach == "beam_search":
    s = set_key(s, "beam_width", bw)

p.write_text(s)
PY
}

run_cmd() {
  local label=$1 logfile=$2
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

set_system_prompt_preset

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] TASK_RESUME_START dataset=aime n=${N}"
  echo "[$(ts)] model=${MODEL}"
  echo "[$(ts)] seed=${SEED}"
  echo "[$(ts)] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} max_tokens=${MAX_TOKENS}"
  echo "================================================================"
} | tee -a "$MASTER_LOG"

set_recipe "recipes/best-of-n.yaml" "best_of_n" "$N" "false" "none"
run_cmd "BON_TRACE_VLLM ds=aime n=${N}" \
  "$LOG_DIR/bon_aime_trace_vllm_n${N}.log" \
  "$PY" scripts/best_of_n_task_trace.py recipes/best-of-n.yaml

set_recipe "recipes/best-of-n.yaml" "best_of_n" "$N" "true" "dynamic"
run_cmd "BON_REPLAY_OURS ds=aime n=${N}" \
  "$LOG_DIR/bon_aime_replay_ours_n${N}.log" \
  "$PY" scripts/test_time_compute.py recipes/best-of-n.yaml

if [[ "$N" -eq 2 ]]; then
  BW=2
else
  BW=4
fi

set_recipe "recipes/beam-search.yaml" "beam_search" "$N" "false" "none" "$BW"
run_cmd "BS_TRACE_VLLM ds=aime n=${N} bw=${BW}" \
  "$LOG_DIR/bs_aime_trace_vllm_n${N}.log" \
  "$PY" scripts/beam_search_task_trace.py recipes/beam-search.yaml

set_recipe "recipes/beam-search.yaml" "beam_search" "$N" "true" "dynamic" "$BW"
run_cmd "BS_REPLAY_OURS ds=aime n=${N} bw=${BW}" \
  "$LOG_DIR/bs_aime_replay_ours_n${N}.log" \
  "$PY" scripts/beam_search_beam.py recipes/beam-search.yaml

log "TASK_RESUME_DONE dataset=aime n=${N}"
