#!/usr/bin/env bash
# Drives the AIME (AI-MO/aimo-validation-aime, split=train) sweep described in
# AIME_SWEEP.md. Order: all best_of_n n's (trace + replay), then all
# beam_search n's (trace + replay). Per-step logs land in logs/aime/.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Datasets/weights are cached locally; skip hub round-trips.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="${PYTHON:-python}"
LOG_DIR=logs/aime
mkdir -p "$LOG_DIR"

ts() { date '+%F %T'; }

set_recipe_bon() {
  # $1=n  $2=gemm_opt  $3=chunk_size
  local n=$1 gemm=$2 chunk=$3
  "$PY" - <<PY
import re, pathlib
p = pathlib.Path('recipes/best-of-n.yaml')
s = p.read_text()
s = re.sub(r'^n: .*$', 'n: ${n}', s, flags=re.M)
s = re.sub(r'^gemm_opt: .*$', 'gemm_opt: ${gemm}', s, flags=re.M)
s = re.sub(r'^(chunk_size: )\S+(.*)$', r'\1${chunk}\2', s, flags=re.M)
p.write_text(s)
PY
}

set_recipe_beam() {
  # $1=n  $2=beam_width  $3=gemm_opt  $4=chunk_size
  local n=$1 bw=$2 gemm=$3 chunk=$4
  "$PY" - <<PY
import re, pathlib
p = pathlib.Path('recipes/beam-search.yaml')
s = p.read_text()
s = re.sub(r'^n: .*$', 'n: ${n}', s, flags=re.M)
s = re.sub(r'^beam_width: .*$', 'beam_width: ${bw}', s, flags=re.M)
s = re.sub(r'^gemm_opt: .*$', 'gemm_opt: ${gemm}', s, flags=re.M)
s = re.sub(r'^(chunk_size: )\S+(.*)$', r'\1${chunk}\2', s, flags=re.M)
p.write_text(s)
PY
}

NS=(2 4 8 16 32 64 128)

# ---------------- Part A: best_of_n ----------------
for n in "${NS[@]}"; do
  echo "[$(ts)] BON_TRACE_START n=$n"
  set_recipe_bon "$n" "false" "none"
  "$PY" scripts/best_of_n_task_trace.py recipes/best-of-n.yaml \
      > "$LOG_DIR/bon_trace_n${n}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] BON_TRACE_FAIL n=$n rc=$rc"
    exit $rc
  fi
  echo "[$(ts)] BON_TRACE_OK n=$n"

  echo "[$(ts)] BON_REPLAY_START n=$n"
  set_recipe_bon "$n" "true" "dynamic"
  "$PY" scripts/test_time_compute.py recipes/best-of-n.yaml \
      > "$LOG_DIR/bon_replay_n${n}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] BON_REPLAY_FAIL n=$n rc=$rc"
    exit $rc
  fi
  echo "[$(ts)] BON_REPLAY_OK n=$n"
done

# ---------------- Part B: beam_search ----------------
for n in "${NS[@]}"; do
  if [[ $n -eq 2 ]]; then bw=2; else bw=4; fi

  echo "[$(ts)] BS_TRACE_START n=$n bw=$bw"
  set_recipe_beam "$n" "$bw" "false" "none"
  "$PY" scripts/beam_search_task_trace.py recipes/beam-search.yaml \
      > "$LOG_DIR/bs_trace_n${n}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] BS_TRACE_FAIL n=$n rc=$rc"
    exit $rc
  fi
  echo "[$(ts)] BS_TRACE_OK n=$n"

  echo "[$(ts)] BS_REPLAY_START n=$n bw=$bw"
  set_recipe_beam "$n" "$bw" "true" "dynamic"
  "$PY" scripts/beam_search_beam.py recipes/beam-search.yaml \
      > "$LOG_DIR/bs_replay_n${n}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] BS_REPLAY_FAIL n=$n rc=$rc"
    exit $rc
  fi
  echo "[$(ts)] BS_REPLAY_OK n=$n"
done

echo "[$(ts)] AIME_SWEEP_COMPLETE"
