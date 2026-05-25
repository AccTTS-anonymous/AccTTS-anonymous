#!/usr/bin/env bash
# Sequentially runs step4, step5, step6 of the Llama-3.2-1B sweep.
# Stops on first non-zero exit. Per-step master logs land in logs/.
# Chain status goes to logs/llama1b_chain456_master.log.
#
# Usage: HF_TOKEN=... bash scripts/run_llama1b_sweep_chain456.sh

set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

: "${HF_TOKEN:?HF_TOKEN must be exported}"

ts() { date '+%F %T'; }
LOG=logs/llama1b_chain456_master.log

echo "[$(ts)] CHAIN456_START" >> "$LOG"

for step in step4 step5 step6; do
  echo "[$(ts)] CHAIN456_RUN $step" >> "$LOG"
  bash scripts/run_llama1b_sweep.sh "$step" > "logs/llama1b_${step}_master.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] CHAIN456_FAIL $step rc=$rc" >> "$LOG"
    exit $rc
  fi
  echo "[$(ts)] CHAIN456_OK   $step" >> "$LOG"
done

echo "[$(ts)] CHAIN456_COMPLETE all phases done" >> "$LOG"
