# Ablation Study Log — `onlyGEMM` variant (best_of_n, QWen2.5-1.5B)

This log records every action taken for the `onlyGEMM` ablation defined in
`ablation_study.md`.

## Pre-flight checks

- `recipes/best-of-n.yaml`: model `Qwen/QWen2.5-1.5B-Instruct`, `approach: best_of_n`,
  `gpu_memory_utilization: 0.9`, `disable_prm: true`, `gemm_opt: true`,
  `chunk_size: none`. OK.
- `src/sal/utils/data.py`: `_method_suffix()` already has the
  `(gemm_opt=true, chunk_size=none) -> "onlyGEMM"` branch. OK (uncommitted edit).
- `src/sal/config.py`: `_SYSTEM_PROMPT_PRESET = "MATH500"`. OK for Part 1.
- Trace files confirmed for all `n ∈ {2,4,8,16,32,64,128}`:
  - `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beams={n}` — present.
  - `data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/beams={n}` — present.
- `test_time_compute.py` resolves the trace via `trace_path_for_config()` to
  `data/{model_path}/{dataset_short}/beams={n}`; `save_dataset()` writes outputs
  to `data/{model_path}/{dataset_short}/`. Both confirmed.

## Actions

| # | Action | Status |
|---|--------|--------|
| 1 | Created this log file. | OK |
| 2 | Verified `_method_suffix()` onlyGEMM branch + recipe `gemm_opt: true`, `chunk_size: none`; set recipe `n: 2`. `git commit` (Add onlyGEMM method suffix and enable gemm_opt). | OK |
| 3 | **Part 1 MATH-500** — recipe `dataset_name: HuggingFaceH4/MATH-500`, `dataset_split: test`, `max_tokens: 2048`; `config.py` preset `MATH500`. | OK |
| 4 | n=2: `python scripts/test_time_compute.py recipes/best-of-n.yaml` — trace `beams=2` (1000 entries), no suffix warning. Outputs `best_of_n_completions_onlyGEMM_n2.jsonl` + `best_of_n_request_timings_onlyGEMM_n2.csv`. | SUCCESS |
| 5 | n=4: same command — trace `beams=4` (2000 entries). Outputs `..._onlyGEMM_n4.jsonl` + `..._onlyGEMM_n4.csv`. | SUCCESS |
| 6 | n=8: same command — trace `beams=8` (4000 entries). Outputs `..._onlyGEMM_n8.jsonl` + `..._onlyGEMM_n8.csv`. | SUCCESS |
| 7 | n=16: same command — trace `beams=16` (8000 entries). Outputs `..._onlyGEMM_n16.jsonl` + `..._onlyGEMM_n16.csv`. | SUCCESS |
| 8 | n=32: same command — trace `beams=32` (16000 entries). Outputs `..._onlyGEMM_n32.jsonl` + `..._onlyGEMM_n32.csv`. | SUCCESS |
| 9 | n=64: same command — trace `beams=64` (32000 entries). Outputs `..._onlyGEMM_n64.jsonl` + `..._onlyGEMM_n64.csv`. | SUCCESS |
| 10 | n=128: same command — trace `beams=128` (64000 entries). Outputs `..._onlyGEMM_n128.jsonl` + `..._onlyGEMM_n128.csv`. A `sal.utils.math.TimeoutException` traceback appears in the log — benign: the scorer's signal-based timeout handler fired inside a `datasets` Arrow `__del__` during GC; run reached "Done", both files saved, scoring continued. | SUCCESS |
| 11 | Part 1 MATH-500 complete. Verified all 7 `n`: completions = 500 rows each, timings CSV = n×500 + 1 header row each. | OK |
| 12 | **Part 2 AIME** — `config.py` preset `MATH500` → `AIME`; `git commit` (Set system prompt preset to AIME). Recipe: `dataset_name: AI-MO/aimo-validation-aime`, `dataset_split: train`, `max_tokens: 4096`. | OK |
| 13 | n=2: `python scripts/test_time_compute.py recipes/best-of-n.yaml` — trace `beams=2` (180 entries = 90 problems × 2). Outputs `..._onlyGEMM_n2.jsonl` + `..._onlyGEMM_n2.csv`. | SUCCESS |
| 14 | n=4: same command — trace `beams=4` (360 entries). Outputs `..._onlyGEMM_n4.jsonl` + `..._onlyGEMM_n4.csv`. | SUCCESS |
| 15 | n=8: same command — trace `beams=8` (720 entries). Outputs `..._onlyGEMM_n8.jsonl` + `..._onlyGEMM_n8.csv`. | SUCCESS |
| 16 | n=16: same command — trace `beams=16` (1440 entries). Outputs `..._onlyGEMM_n16.jsonl` + `..._onlyGEMM_n16.csv`. | SUCCESS |
| 17 | n=32: same command — trace `beams=32` (2880 entries). Outputs `..._onlyGEMM_n32.jsonl` + `..._onlyGEMM_n32.csv`. | SUCCESS |
| 18 | n=64: same command — trace `beams=64` (5760 entries). Outputs `..._onlyGEMM_n64.jsonl` + `..._onlyGEMM_n64.csv`. | SUCCESS |
| 19 | n=128: same command — trace `beams=128` (11520 entries). Outputs `..._onlyGEMM_n128.jsonl` + `..._onlyGEMM_n128.csv`. | SUCCESS |
| 20 | Part 2 AIME complete. Verified all 7 `n`: completions = 90 rows each, timings CSV = n×90 + 1 header row each. | OK |
| 21 | Restored `config.py` preset `AIME` → `MATH500`; `git commit` (Restore system prompt preset to MATH500). | OK |

## Summary

Both phases of the `onlyGEMM` ablation finished with no skipped/failed test cases.

- **Part 1 — MATH-500** (`max_tokens: 2048`, preset `MATH500`): all
  `n ∈ {2,4,8,16,32,64,128}` succeeded. 500 problems per `n`.
- **Part 2 — AIME** (`max_tokens: 4096`, preset `AIME`): all
  `n ∈ {2,4,8,16,32,64,128}` succeeded. 90 problems per `n`.
- All 28 output files (`best_of_n_completions_onlyGEMM_n{n}.jsonl` +
  `best_of_n_request_timings_onlyGEMM_n{n}.csv`, two datasets × 7 `n`) written
  under `data/Qwen/QWen2.5-1.5B-Instruct/{MATH-500,aimo-validation-aime}/`.
- `_method_suffix()` produced the `onlyGEMM` suffix on every run — no
  "does not match any of the named methods" warning was logged.
- Every run loaded its `beams={n}` trace (force_beam_gen active): forced
  `(problem, beam)` counts equal `n × num_problems` in all 14 runs.

### Notes / non-fatal observations

- **MATH-500 n=128**: a `sal.utils.math.TimeoutException` traceback appears in
  `ablation_runs/math_onlyGEMM_n128.log`. It is benign — the scorer's
  signal-based timeout handler (`src/sal/utils/math.py:37`) fired inside a
  `datasets` Arrow object `__del__` during garbage collection. The run reached
  "Done ??!", both output files were saved, and weighted/majority scoring
  completed normally afterward. No data lost; nothing skipped.

### Code changes committed

- `e51cfef` — Add `onlyGEMM` method suffix (`src/sal/utils/data.py`) and enable
  `gemm_opt` (`recipes/best-of-n.yaml`).
- `487c4ed` — Set `_SYSTEM_PROMPT_PRESET = "AIME"` (`src/sal/config.py`).
- `963f75c` — Restore `_SYSTEM_PROMPT_PRESET = "MATH500"` (`src/sal/config.py`).

Per-run stdout logs are kept under `ablation_runs/`.
