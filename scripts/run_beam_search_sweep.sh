#!/usr/bin/env bash
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Weights/datasets are cached locally; skip hub round-trips that timed out on fresh process startup.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="${PYTHON:-python}"
mkdir -p logs

ts() { date '+%F %T'; }

set_recipe() {
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

# n=2 trace already completed in a prior pass — skip it on resume.

NS=(2 4 8 16 32 64 128)

for n in "${NS[@]}"; do
  if [[ $n -eq 2 ]]; then bw=2; else bw=4; fi

  # n=2 trace already ran; everything else needs trace + replay.
  if [[ $n -ne 2 ]]; then
    echo "[$(ts)] PHASE_START trace n=$n bw=$bw"
    set_recipe "$n" "$bw" "false" "none"
    "$PY" scripts/beam_search_task_trace.py recipes/beam-search.yaml > "logs/trace_n${n}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(ts)] PHASE_FAIL trace n=$n rc=$rc"
      exit $rc
    fi
    echo "[$(ts)] PHASE_OK trace n=$n"
  fi

  echo "[$(ts)] PHASE_START replay n=$n bw=$bw"
  set_recipe "$n" "$bw" "true" "dynamic"
  "$PY" scripts/beam_search_beam.py recipes/beam-search.yaml > "logs/replay_n${n}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] PHASE_FAIL replay n=$n rc=$rc"
    exit $rc
  fi
  echo "[$(ts)] PHASE_OK replay n=$n"
done

echo "[$(ts)] SWEEP_COMPLETE"
