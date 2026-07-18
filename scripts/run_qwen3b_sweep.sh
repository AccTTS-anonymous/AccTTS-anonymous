#!/usr/bin/env bash
# Drives the QWen2.5-3B-Instruct sweep described in QWEN3B_SWEEP.md.
#
# Order (Steps 3-6):
#   Step 3: MATH-500  best_of_n   (n=2,4,8,16,32,64)
#   Step 4: MATH-500  beam_search (n=2 uses bw=2; others bw=4)
#   Step 5: AIME      best_of_n
#   Step 6: AIME      beam_search
#
# Per-step logs land in logs/qwen3b/. Master status is printed to stdout
# (caller is expected to redirect to a log).
#
# Resumability: the master log lines (TRACE_OK / REPLAY_OK) are written as
# events are completed. If you re-launch this script, the per-step logs land
# under the same paths and will be overwritten; expected artifacts under
# data/Qwen/QWen2.5-3B-Instruct/<dataset>/ should be checked by hand for
# partial outputs before relaunching from a fresh state.

set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# HF caches live on the data volume.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
# We do not set HF_HUB_OFFLINE because the model dir is keyed on the
# case-sensitive name "QWen2.5-3B-Instruct" while the canonical HF id is
# "Qwen2.5-3B-Instruct"; offline mode skips the case-insensitive resolution.
# A network round-trip per run is acceptable (the actual weights/dataset are
# resolved from cache once the case is reconciled).

PY="${PYTHON:-python}"
LOG_DIR=logs/qwen3b
mkdir -p "$LOG_DIR"

ts() { date '+%F %T'; }

set_system_prompt_preset() {
  # $1 = 'AIME' | 'MATH500'
  local preset=$1
  "$PY" - "$preset" <<'PY'
import re, sys, pathlib
preset = sys.argv[1]
assert preset in ("AIME", "MATH500"), preset
p = pathlib.Path('src/sal/config.py')
s = p.read_text()
s = re.sub(r'^_SYSTEM_PROMPT_PRESET = .*$', f'_SYSTEM_PROMPT_PRESET = "{preset}"', s, flags=re.M)
p.write_text(s)
PY
}

# Recipe editors. Each uses python's regex.MULTILINE because YAML keys can be
# preceded by comments (commented-out dataset lines) and we only want to
# update active keys.

set_recipe_bon() {
  # $1=n  $2=gemm_opt  $3=chunk_size  $4=dataset_state ('aime'|'math500')
  local n=$1 gemm=$2 chunk=$3 ds=$4
  "$PY" - "$n" "$gemm" "$chunk" "$ds" <<'PY'
import re, sys, pathlib
n, gemm, chunk, ds = sys.argv[1:5]
p = pathlib.Path('recipes/best-of-n.yaml')
s = p.read_text()
s = re.sub(r'^n: .*$', f'n: {n}', s, flags=re.M)
s = re.sub(r'^gemm_opt: .*$', f'gemm_opt: {gemm}', s, flags=re.M)
s = re.sub(r'^(chunk_size: )\S+(.*)$', rf'\g<1>{chunk}\g<2>', s, flags=re.M)
if ds == 'aime':
    # Uncomment dataset lines if present.
    s = re.sub(r'^#\s*(dataset_name: AI-MO/aimo-validation-aime)\s*$', r'\1', s, flags=re.M)
    s = re.sub(r'^#\s*(dataset_split: train)\s*$', r'\1', s, flags=re.M)
else:
    # Comment dataset lines so Config defaults (MATH-500) take over.
    s = re.sub(r'^(dataset_name: AI-MO/aimo-validation-aime)\s*$', r'# \1', s, flags=re.M)
    s = re.sub(r'^(dataset_split: train)\s*$', r'# \1', s, flags=re.M)
p.write_text(s)
PY
}

set_recipe_beam() {
  # $1=n  $2=beam_width  $3=gemm_opt  $4=chunk_size  $5=dataset_state ('aime'|'math500')
  local n=$1 bw=$2 gemm=$3 chunk=$4 ds=$5
  "$PY" - "$n" "$bw" "$gemm" "$chunk" "$ds" <<'PY'
import re, sys, pathlib
n, bw, gemm, chunk, ds = sys.argv[1:6]
p = pathlib.Path('recipes/beam-search.yaml')
s = p.read_text()
s = re.sub(r'^n: .*$', f'n: {n}', s, flags=re.M)
s = re.sub(r'^beam_width: .*$', f'beam_width: {bw}', s, flags=re.M)
s = re.sub(r'^gemm_opt: .*$', f'gemm_opt: {gemm}', s, flags=re.M)
s = re.sub(r'^(chunk_size: )\S+(.*)$', rf'\g<1>{chunk}\g<2>', s, flags=re.M)
if ds == 'aime':
    s = re.sub(r'^#\s*(dataset_name: AI-MO/aimo-validation-aime)\s*$', r'\1', s, flags=re.M)
    s = re.sub(r'^#\s*(dataset_split: train)\s*$', r'\1', s, flags=re.M)
else:
    s = re.sub(r'^(dataset_name: AI-MO/aimo-validation-aime)\s*$', r'# \1', s, flags=re.M)
    s = re.sub(r'^(dataset_split: train)\s*$', r'# \1', s, flags=re.M)
p.write_text(s)
PY
}

NS=(2 4 8 16 32 64)

run_bon_phase() {
  # $1 = dataset_state ('math500'|'aime')
  local ds=$1
  local tag preset
  if [[ "$ds" == "aime" ]]; then tag="aime"; preset="AIME"; else tag="math500"; preset="MATH500"; fi
  set_system_prompt_preset "$preset"
  echo "[$(ts)] SYSTEM_PROMPT_PRESET=$preset (bon phase, ds=$ds)"
  for n in "${NS[@]}"; do
    echo "[$(ts)] BON_TRACE_START   ds=$ds n=$n"
    set_recipe_bon "$n" "false" "none" "$ds"
    "$PY" scripts/best_of_n_task_trace.py recipes/best-of-n.yaml \
        > "$LOG_DIR/bon_${tag}_trace_n${n}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(ts)] BON_TRACE_FAIL    ds=$ds n=$n rc=$rc"
      exit $rc
    fi
    echo "[$(ts)] BON_TRACE_OK      ds=$ds n=$n"

    echo "[$(ts)] BON_REPLAY_START  ds=$ds n=$n"
    set_recipe_bon "$n" "true" "dynamic" "$ds"
    "$PY" scripts/test_time_compute.py recipes/best-of-n.yaml \
        > "$LOG_DIR/bon_${tag}_replay_n${n}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(ts)] BON_REPLAY_FAIL   ds=$ds n=$n rc=$rc"
      exit $rc
    fi
    echo "[$(ts)] BON_REPLAY_OK     ds=$ds n=$n"
  done
}

run_bs_phase() {
  # $1 = dataset_state ('math500'|'aime')
  local ds=$1
  local tag preset
  if [[ "$ds" == "aime" ]]; then tag="aime"; preset="AIME"; else tag="math500"; preset="MATH500"; fi
  set_system_prompt_preset "$preset"
  echo "[$(ts)] SYSTEM_PROMPT_PRESET=$preset (bs phase, ds=$ds)"
  for n in "${NS[@]}"; do
    if [[ $n -eq 2 ]]; then bw=2; else bw=4; fi

    echo "[$(ts)] BS_TRACE_START    ds=$ds n=$n bw=$bw"
    set_recipe_beam "$n" "$bw" "false" "none" "$ds"
    "$PY" scripts/beam_search_task_trace.py recipes/beam-search.yaml \
        > "$LOG_DIR/bs_${tag}_trace_n${n}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(ts)] BS_TRACE_FAIL     ds=$ds n=$n rc=$rc"
      exit $rc
    fi
    echo "[$(ts)] BS_TRACE_OK       ds=$ds n=$n"

    echo "[$(ts)] BS_REPLAY_START   ds=$ds n=$n bw=$bw"
    set_recipe_beam "$n" "$bw" "true" "dynamic" "$ds"
    "$PY" scripts/beam_search_beam.py recipes/beam-search.yaml \
        > "$LOG_DIR/bs_${tag}_replay_n${n}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(ts)] BS_REPLAY_FAIL    ds=$ds n=$n rc=$rc"
      exit $rc
    fi
    echo "[$(ts)] BS_REPLAY_OK      ds=$ds n=$n"
  done
}

PHASES="${1:-all}"

case "$PHASES" in
  step3|math500_bon)
    echo "[$(ts)] PHASE_START step3 (MATH-500 best_of_n)"
    run_bon_phase math500
    echo "[$(ts)] PHASE_DONE  step3"
    ;;
  step4|math500_bs)
    echo "[$(ts)] PHASE_START step4 (MATH-500 beam_search)"
    run_bs_phase math500
    echo "[$(ts)] PHASE_DONE  step4"
    ;;
  step5|aime_bon)
    echo "[$(ts)] PHASE_START step5 (AIME best_of_n)"
    run_bon_phase aime
    echo "[$(ts)] PHASE_DONE  step5"
    ;;
  step6|aime_bs)
    echo "[$(ts)] PHASE_START step6 (AIME beam_search)"
    run_bs_phase aime
    echo "[$(ts)] PHASE_DONE  step6"
    ;;
  all)
    echo "[$(ts)] PHASE_START all"
    run_bon_phase math500
    run_bs_phase math500
    run_bon_phase aime
    run_bs_phase aime
    echo "[$(ts)] PHASE_DONE  all"
    ;;
  *)
    echo "Usage: $0 [step3|step4|step5|step6|all]" >&2
    exit 2
    ;;
esac

echo "[$(ts)] QWEN3B_SWEEP_COMPLETE"
