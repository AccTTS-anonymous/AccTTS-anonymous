#!/usr/bin/env bash
# Waits for the in-flight step3 driver (passed as $1) to exit cleanly, then
# runs step4, step5, step6 sequentially. Each step writes its own master log
# (logs/qwen3b_step{N}_master.log) and per-step logs land in logs/qwen3b/.
#
# Usage: run_qwen3b_sweep_chain.sh <pid-of-step3-driver>
#
# Exits early if the upstream driver dies non-zero (we infer this by looking
# for the matching "PHASE_DONE step3" line in its master log).

set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

UPSTREAM_PID=${1:?usage: $0 <upstream_pid>}
UPSTREAM_LOG=logs/qwen3b_step3_master.log

ts() { date '+%F %T'; }
LOG=logs/qwen3b_chain_master.log

echo "[$(ts)] CHAIN_START waiting on upstream PID=$UPSTREAM_PID" >> "$LOG"

# Block until the upstream driver exits.
while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
  sleep 60
done

if ! grep -q "PHASE_DONE  step3" "$UPSTREAM_LOG" 2>/dev/null; then
  echo "[$(ts)] CHAIN_ABORT upstream did not complete step3 cleanly; see $UPSTREAM_LOG" >> "$LOG"
  exit 1
fi
echo "[$(ts)] CHAIN_UPSTREAM_OK step3 completed; starting step4" >> "$LOG"

for step in step4 step5 step6; do
  echo "[$(ts)] CHAIN_RUN $step" >> "$LOG"
  bash scripts/run_qwen3b_sweep.sh "$step" > "logs/qwen3b_${step}_master.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] CHAIN_FAIL $step rc=$rc" >> "$LOG"
    exit $rc
  fi
  echo "[$(ts)] CHAIN_OK   $step" >> "$LOG"
done

echo "[$(ts)] CHAIN_COMPLETE all phases done" >> "$LOG"
