# AIME Sweep Log

Per AIME_SWEEP.md instructions, this file records every action taken during the
AIME sweep (best_of_n then beam_search, n ∈ {2, 4, 8, 16, 32, 64, 128}) along
with success/fail status and any code or recipe modifications.

Working dir: `/path/to/AccTTS`
Conda env:   `sal_new_vllm` (active)
Env vars:    `VLLM_USE_V1=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`,
             `VLLM_ENABLE_V1_MULTIPROCESSING=0`

---

## 2026-05-10 — Setup

- **OK** Verified working dir, conda env, and env vars per spec.
- **OK** Confirmed `recipes/best-of-n.yaml` and `recipes/beam-search.yaml`
  already declare `dataset_name: AI-MO/aimo-validation-aime` and
  `dataset_split: train` (uncommitted change, will be included in the sweep
  commit).
- **OK** GPU 0 (RTX 4090, 24 GB) idle.
- **OK** Output dir `data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`
  does not yet exist; will be created by the trace runs.

## Actions

### 2. Launched sweep (first attempt — failed)
- **OK** Started `nohup bash scripts/run_aime_sweep.sh > logs/aime_sweep_master.log 2>&1 &`
  at 2026-05-10 20:34:44 (PID 802785). Master log: `logs/aime_sweep_master.log`;
  per-step logs: `logs/aime/{bon,bs}_{trace,replay}_n{N}.log`.
- **FAIL** `BON_TRACE_FAIL n=2 rc=1` at 2026-05-10 20:35:26.
  - Root cause: `HF_HUB_OFFLINE=1` (carried over from
    `run_beam_search_sweep.sh`) blocked the AIME dataset fetch from the Hub —
    `AI-MO/aimo-validation-aime` was not in `/path/to/hf_cache/datasets/`.
    Model weights were already cached so the model loaded fine; only the
    dataset lookup tripped on offline mode.
  - Fix: pre-cached the dataset by running `load_dataset("AI-MO/aimo-validation-aime",
    split="train")` with offline flags stripped (90 rows, columns
    `[id, problem, solution, answer, url]`). No code change to the sweep
    script — once the dataset lives in `~/.cache/huggingface/datasets/...`,
    `HF_HUB_OFFLINE=1` is harmless because `load_dataset` will hit the cache
    first.

### 3. Re-launched sweep (PID 811985)
- 2026-05-10 20:39:32  BON_TRACE_START  n=2
- 2026-05-10 20:52:19  **OK** BON_TRACE  n=2  (~12.8 min, artifacts: `beams=2`, `beams=2.csv`, `beams=2_timing.csv`, `best_of_n_completions_vllm_n2.jsonl`, `best_of_n_request_timings_vllm_n2.csv`)
- 2026-05-10 20:52:19  BON_REPLAY_START n=2
- 2026-05-10 21:03:26  **OK** BON_REPLAY n=2  (~11.1 min, artifacts: `best_of_n_completions_ours_n2.jsonl`, `best_of_n_request_timings_ours_n2.csv`)
- 2026-05-10 21:03:26  BON_TRACE_START  n=4
- 2026-05-10 21:21:30  **OK** BON_TRACE  n=4  (~18.1 min)
- 2026-05-10 21:21:30  BON_REPLAY_START n=4
- 2026-05-10 21:35:54  **OK** BON_REPLAY n=4  (~14.4 min)
- 2026-05-10 21:35:54  BON_TRACE_START  n=8
- 2026-05-10 21:58:35  **OK** BON_TRACE  n=8  (~22.7 min)
- 2026-05-10 21:58:35  BON_REPLAY_START n=8
- 2026-05-10 22:15:54  **OK** BON_REPLAY n=8  (~17.3 min)
- 2026-05-10 22:15:55  BON_TRACE_START  n=16
- 2026-05-10 22:44:20  **OK** BON_TRACE  n=16 (~28.4 min)
- 2026-05-10 22:44:20  BON_REPLAY_START n=16
- 2026-05-10 23:05:16  **OK** BON_REPLAY n=16 (~20.9 min)
- 2026-05-10 23:05:16  BON_TRACE_START  n=32
- 2026-05-10 23:42:58  **OK** BON_TRACE  n=32 (~37.7 min)
- 2026-05-10 23:42:58  BON_REPLAY_START n=32
- 2026-05-11 00:10:28  **OK** BON_REPLAY n=32 (~27.5 min)
- 2026-05-11 00:10:28  BON_TRACE_START  n=64
- 2026-05-11 00:57:02  **OK** BON_TRACE  n=64 (~46.6 min)
- 2026-05-11 00:57:02  BON_REPLAY_START n=64
- 2026-05-11 01:30:28  **OK** BON_REPLAY n=64 (~33.4 min)
- 2026-05-11 01:30:28  BON_TRACE_START  n=128
- 2026-05-11 02:25:34  **OK** BON_TRACE  n=128 (~55.1 min)
- 2026-05-11 02:25:34  BON_REPLAY_START n=128
- 2026-05-11 03:06:57  **OK** BON_REPLAY n=128 (~41.4 min)

**Part A (best_of_n) complete — all 7 n's traced + replayed cleanly. Total: ~6h27m.**

- 2026-05-11 03:06:57  BS_TRACE_START  n=2 bw=2
- 2026-05-11 03:21:32  **OK** BS_TRACE  n=2 (~14.6 min)
- 2026-05-11 03:21:32  BS_REPLAY_START n=2 bw=2
- 2026-05-11 03:33:53  **OK** BS_REPLAY n=2 (~12.4 min)
- 2026-05-11 03:33:53  BS_TRACE_START  n=4 bw=4
- 2026-05-11 03:52:15  **OK** BS_TRACE  n=4 (~18.4 min)
- 2026-05-11 03:52:15  BS_REPLAY_START n=4 bw=4
- 2026-05-11 04:07:36  **OK** BS_REPLAY n=4 (~15.4 min)
- 2026-05-11 04:07:36  BS_TRACE_START  n=8 bw=4
- 2026-05-11 04:48:16  **OK** BS_TRACE  n=8 (~40.7 min)
- 2026-05-11 04:48:16  BS_REPLAY_START n=8 bw=4
- 2026-05-11 05:15:06  **OK** BS_REPLAY n=8 (~26.8 min)
- 2026-05-11 05:15:06  BS_TRACE_START  n=16 bw=4
- 2026-05-11 06:28:32  **OK** BS_TRACE  n=16 (~73.4 min)
- 2026-05-11 06:28:32  BS_REPLAY_START n=16 bw=4
- 2026-05-11 07:05:37  **OK** BS_REPLAY n=16 (~37.1 min)
- 2026-05-11 07:05:37  BS_TRACE_START  n=32 bw=4
- 2026-05-11 09:19:35  **OK** BS_TRACE  n=32 (~134.0 min)
- 2026-05-11 09:19:35  BS_REPLAY_START n=32 bw=4
- 2026-05-11 10:21:22  **OK** BS_REPLAY n=32 (~61.8 min)
- 2026-05-11 10:21:22  BS_TRACE_START  n=64 bw=4
- 2026-05-11 14:10:55  **OK** BS_TRACE  n=64 (~229.6 min)
- 2026-05-11 14:10:55  BS_REPLAY_START n=64 bw=4
- 2026-05-11 15:38:55  **OK** BS_REPLAY n=64 (~88.0 min)
- 2026-05-11 15:38:55  BS_TRACE_START  n=128 bw=4
- 2026-05-11 21:37:44  **OK** BS_TRACE  n=128 (~358.8 min)
- 2026-05-11 21:37:44  BS_REPLAY_START n=128 bw=4
- 2026-05-11 23:46:14  **OK** BS_REPLAY n=128 (~128.5 min)
- 2026-05-11 23:46:14  **AIME_SWEEP_COMPLETE**

**Part B (beam_search) complete — all 7 n's traced + replayed cleanly. Total: ~20h39m.**

### 4. Final summary
- **OK** Both methods (best_of_n, beam_search) finished for all
  `n ∈ {2, 4, 8, 16, 32, 64, 128}`.
- Total wall time: ~27h11m (start 2026-05-10 20:39:32, end 2026-05-11 23:46:14).
- No problems were skipped; no OOM/hang; no code changes needed beyond the
  recipe `dataset_name`/`dataset_split` additions present at session start and
  the new sweep driver script.
- All expected artifacts present in
  `data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`:
  - 7× `beams={n}` trace literals + `.csv` + `_timing.csv`
  - 7× `best_of_n_completions_{vllm,ours}_n{n}.jsonl`
  - 7× `best_of_n_request_timings_{vllm,ours}_n{n}.csv`
  - 7× `beam_search_n{n}` trace literals + `.csv` + `_timing.csv`
  - 7× `beam_search_request_timings_{vllm,ours}_n{n}.csv`
  - 7× `beam_search_per_step_ours_n{n}.csv`

- **Note** AIME_SWEEP.md line 134 lists `beam_search_per_step_vllm_n{n}.csv`,
  but only `scripts/beam_search_beam.py` writes per-step CSVs — and only with
  the suffix derived from `(gemm_opt, chunk_size)` of *its own* run. The
  trace script (`beam_search_task_trace.py`) writes the request-level `_vllm`
  timing CSV but no per-step CSV. To produce `beam_search_per_step_vllm_n{n}.csv`
  you would need to re-run `beam_search_beam.py` with `gemm_opt=false,
  chunk_size=none` against the existing trace. The MATH-500 directory has
  only a partial set of these `_vllm` per-step files (n=2,4,8), suggesting
  the same pattern there. **Not produced for AIME** — flag in case the
  downstream analysis needs them.

### 1. Created sweep driver script
- **OK** Wrote `scripts/run_aime_sweep.sh`. Mirrors
  `scripts/run_beam_search_sweep.sh` but covers both methods on AIME, in the
  order required by AIME_SWEEP.md (best_of_n full sweep first, then beam_search
  full sweep). For each `n`, the script edits the recipe in-place and runs
  trace then replay; per-step stdout/stderr land in `logs/aime/`. The
  beam_search loop sets `beam_width: 2` only when `n == 2`.

</content>
</invoke>