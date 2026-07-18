# Beam Search Sweep Log

Started: 2026-05-09

## Task
Per BEAM_SEARCH.md instructions, sweep beam_search on MATH-500. User-modified scope:
- For n in [2, 4, 8, 16, 32, 64]: run **ours** replay only (traces and vllm timings already exist).
- For n=128: run **both** trace+vllm and ours replay.

## Pre-existing files in `$REPO_ROOT/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/`
- `beam_search_n{2,4,8,16,32,64}` (traces)
- `beam_search_n{2,4,8,16,32,64}.csv`
- `beam_search_n{2,4,8,16,32,64}_timing.csv`
- `beam_search_request_timings_vllm_n{2,4,8,16,32,64}.csv`
Missing: `beam_search_request_timings_ours_n{...}.csv` for all n; everything for n=128.

## Environment
- conda env: `sal_new_vllm`
- VLLM_USE_V1=1, VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1, VLLM_ENABLE_V1_MULTIPROCESSING=0
- cwd: `$REPO_ROOT`

## Actions

- 10:24 — Started n=2 ours replay (recipe: n=2, beam_width=2, gemm_opt=true, chunk_size=dynamic). **FAIL: OOM** because PID 2690311 was occupying 22 GB on the GPU.
- 10:25 — User authorized killing PID 2690311 (a stray `beam_search_task_trace.py` started earlier at 10:16 against `recipes/beam-search.yaml`). Killed; GPU freed (24 GB free).
- 10:26 — Restarted n=2 ours replay. **OK** (11:00). Output: `beam_search_request_timings_ours_n2.csv` (66 KB).
- 11:00 — Recipe edited for n=4 (n=4, beam_width=4, gemm_opt=true, chunk_size=dynamic). n=4 ours replay started.
- 11:42 — n=4 ours **OK**. Output: `beam_search_request_timings_ours_n4.csv` (130 KB).
- 11:42 — Recipe edited for n=8 (n=8, beam_width=4). n=8 ours replay started.
- 12:55 — n=8 ours **OK**. Output: `beam_search_request_timings_ours_n8.csv`.
- 12:55 — Recipe edited for n=16. n=16 ours replay started.
- 14:47 — n=16 ours **OK**. Output: `beam_search_request_timings_ours_n16.csv`.
- 14:47 — Recipe edited for n=32. n=32 ours replay started.
- ~14:50 — **User issued `stop`.** Killed PIDs 3004003/3004416/3005000 (the n=32 ours replay). GPU freed.

## Sweep status at stop
- Completed (ours): n=2, 4, 8, 16. Output CSVs present in `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beam_search_request_timings_ours_n{2,4,8,16}.csv`.
- Not run: n=32, n=64, n=128 (ours), n=128 (trace+vllm).
- Recipe currently: n=32, beam_width=4, gemm_opt=true, chunk_size=dynamic.

---

## Session 2: 2026-05-09 (resumed)

### Pre-flight check
- Top-level `MATH-500/` only contains the trace literals `beam_search_n{2,4,8,16,32,64}` — no per-step CSVs, no `request_timings_ours_*` outputs at top level. Prior session's "completed" outputs apparently moved into subdirs (`beam_search/`, `beam_search_more_than_n_beams_total_wall_time/`, `beam_search_exact_n_beams_max_gen_time/`). The doc-spec output location (top of MATH-500) is empty. No `n=128` trace yet.
- Stale process PID 2690487 (`beam_search_task_trace.py`) running 8.5h with 1s CPU, GPU only 49 MiB used. **Killed (kill -9)** — GPU now 24035 MiB free.
- Recipe at session start: `n=4, beam_width=4, gemm_opt=true, chunk_size=dynamic`.

### Plan (per user, full sweep top-level)
For each n ∈ [2, 4, 8, 16, 32, 64]: run vllm replay (`gemm_opt=false, chunk_size=none`) then ours replay (`gemm_opt=true, chunk_size=dynamic`). For n=128: Step 0 trace+vllm via `beam_search_task_trace.py`, then Step 1 ours replay only.

**Order revised mid-sweep**: after n=8, jump to n=128 (Step 0 + ours), then resume n=16, 32, 64.

**Scope revised again 2026-05-10 ~12:05** (per user "After sweeping n=128, please do n=64 and n=32 for our method"): post-n=128, only run **ours** for n=64 then n=32. Drop n=16 entirely; drop vllm replays for n=32 and n=64.

### Actions

- 18:36 — Killed stale PID 2690487. GPU freed.
- 18:55 — Edited recipe: n=2, beam_width=2, gemm_opt=false, chunk_size=none. Launched n=2 vllm replay (`scripts/beam_search_beam.py recipes/beam-search.yaml`, log `logs/sweep/n2_vllm.log`).
- 19:32 — n=2 vllm **OK** (exit 0, ~37 min). Outputs: `beam_search_request_timings_vllm_n2.csv`, `beam_search_per_step_vllm_n2.csv`.
- 19:32 — Edited recipe: gemm_opt=true, chunk_size=dynamic. Launched n=2 ours replay (log `logs/sweep/n2_ours.log`).
- 20:06 — n=2 ours **OK** (exit 0). Output: `beam_search_request_timings_ours_n2.csv`, `beam_search_per_step_ours_n2.csv`.
- 20:06 — Edited recipe: n=4, beam_width=4, gemm_opt=false, chunk_size=none. Launched n=4 vllm replay (log `logs/sweep/n4_vllm.log`).
- 20:51 — n=4 vllm **OK** (exit 0, ~45 min). Output: `beam_search_request_timings_vllm_n4.csv`, `beam_search_per_step_vllm_n4.csv`.
- 20:51 — Edited recipe: gemm_opt=true, chunk_size=dynamic. Launched n=4 ours replay (log `logs/sweep/n4_ours.log`).
- 21:33 — n=4 ours **OK** (exit 0). Output: `beam_search_request_timings_ours_n4.csv`, `beam_search_per_step_ours_n4.csv`.
- 21:33 — Edited recipe: n=8, beam_width=4, gemm_opt=false, chunk_size=none. Launched n=8 vllm replay (log `logs/sweep/n8_vllm.log`).
- 22:41 — n=8 vllm **OK** (exit 0, ~68 min). Output: `beam_search_request_timings_vllm_n8.csv`, `beam_search_per_step_vllm_n8.csv`.
- 22:41 — Edited recipe: gemm_opt=true, chunk_size=dynamic. Launched n=8 ours replay (log `logs/sweep/n8_ours.log`).
- 23:42 — n=8 ours **OK** (exit 0, ~61 min). Output: `beam_search_request_timings_ours_n8.csv`, `beam_search_per_step_ours_n8.csv`.
- 23:42 — Edited recipe: n=128, beam_width=4. Launched n=128 Step 0 (`scripts/beam_search_task_trace.py`, log `logs/sweep/n128_trace.log`). Step 0 internally forces gemm_opt=false/chunk_size=none, so the same run produces both the trace literal and `beam_search_request_timings_vllm_n128.csv`.
- 2026-05-10 12:04 — n=128 trace **OK** (exit 0, ~12.4 h). Outputs: `beam_search_n128` (trace literal, 2.52M entries, 598k completed), `beam_search_n128.csv`, `beam_search_n128_timing.csv`, `beam_search_request_timings_vllm_n128.csv`. (Per the doc, no separate vllm replay needed for n=128 — Step 0's run is the vllm-method experiment.) Mid-run note: server lost connection around 07:45; the python process kept running unaffected and was reattached when the session resumed.
- 12:04 — Edited recipe: gemm_opt=true, chunk_size=dynamic. Launched n=128 ours replay (log `logs/sweep/n128_ours.log`).
- 16:21 — n=128 ours **OK** (exit 0, ~4.3 h). Output: `beam_search_request_timings_ours_n128.csv`, `beam_search_per_step_ours_n128.csv`.
- 16:21 — Edited recipe: n=64. Launched n=64 ours replay (log `logs/sweep/n64_ours.log`).
- 18:25 — n=64 ours **OK** (exit 0, ~2.1 h). Output: `beam_search_request_timings_ours_n64.csv`, `beam_search_per_step_ours_n64.csv`.
- 18:25 — Edited recipe: n=32. Launched n=32 ours replay (log `logs/sweep/n32_ours.log`).
- 20:07 — n=32 ours **OK** (exit 0, ~1.7 h). Output: `beam_search_request_timings_ours_n32.csv`, `beam_search_per_step_ours_n32.csv`.

### Final outputs (under `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/`)
| n | vllm timings | vllm per-step | ours timings | ours per-step |
|---|---|---|---|---|
| 2   | ✓ | ✓ | ✓ | ✓ |
| 4   | ✓ | ✓ | ✓ | ✓ |
| 8   | ✓ | ✓ | ✓ | ✓ |
| 16  | — | — | — | — |  (dropped per user 12:05)
| 32  | — | — | ✓ | ✓ |  (vllm dropped per user 12:05)
| 64  | — | — | ✓ | ✓ |  (vllm dropped per user 12:05)
| 128 | ✓ (from Step 0) | n/a (Step 0 doesn't produce a Step-1-style per-step file for vllm; the `beam_search_n128_timing.csv` per-step file is the equivalent) | ✓ | ✓ |

### Notes / issues
- No code changes were necessary during the sweep. Only `recipes/beam-search.yaml` was repeatedly toggled (`n`, `beam_width`, `gemm_opt`, `chunk_size`). Currently parked at `n=32, beam_width=4, gemm_opt=true, chunk_size=dynamic`.
- One stale Python process (`beam_search_task_trace.py`, PID 2690487) from a prior session was killed at the start; no other failures or skips.
- Mid-sweep the Claude Code session lost connection during the long n=128 trace. The python subprocess kept running independently; reattached cleanly when the session resumed.
- Logs for each step are under `logs/sweep/n{n}_{vllm,ours,trace}.log`.

