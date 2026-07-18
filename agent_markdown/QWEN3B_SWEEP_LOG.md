# QWen2.5-3B-Instruct Sweep Log

Run started: 2026-05-12

This log records every action taken while executing QWEN3B_SWEEP.md, including
code edits, git commits, command invocations, and pass/fail status.

---

## Step 1 — Collect GEMM templates

- **Action:** Edited `scripts/gemm_best_templates_collect.py:24` —
  `COMPILE_LADDER = [1, 2, 4] + list(range(8, 65, 8))`
  (yields `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]`, 11 shapes).
- **Commit:** `1a58b19` — "gemm collect: trim compile ladder to n<=64 for 3B sweep"
- **Status:** edit + commit success.
- **Run:**
  `python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml`
  (launched as PID 3722955, logs at `logs/gemm_collect_3b.log`).

## Step 2 — Profile chunk settings + build LUT (prep)

- **Action:** Edited `scripts/chunk_setting_profile.py:62` —
  `N_BEAMS_LIST = [1, 2, 4] + list(range(8, 65, 8))`.
- **Commit:** `2ef5b97` — "chunk profile: trim n_beams sweep to <=64 for 3B sweep".
- **Action:** Edited `scripts/profile_results_analyze.ipynb` cell `0bd7f9f0`
  `MODEL_PATH = 'Qwen/QWen2.5-3B-Instruct'`.
- **Commit:** `280a6a3` — "profile notebook: set MODEL_PATH to QWen2.5-3B-Instruct".
- **Status:** edits + commits success; profiler run pending Step 1 completion.

### Step 1 — run + verify

- **Run:** `python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml`
  → "Pre-compile finished for 11 shapes: 1..64." (success).
- Cache directory: `~/.cache/vllm/torch_compile_cache/c473fd9511/rank_0_0/`
  (mtime 18:45 on 2026-05-12). Contains `computation_graph.py`,
  `inductor_cache`, `transformed_code.py`, `triton_cache`,
  `vllm_compile_cache.py`. **Verified.**

### Step 2 — profiler run

- **Run:** `python scripts/chunk_setting_profile.py recipes/best-of-n.yaml`
  (PID 3742976; logs `logs/chunk_profile_3b.log`).
- Initial prompt length 305 → start_len rounded to 320.
- Final CSV `chunk_setting_profile_results.csv`: 1701 lines, **0 NaN/OOM rows**.
- Last row recorded: `n_beams=64 prompt_len=4096 FD-64 → 4504 ms`. Sweep
  completed in ~48 min (18:46:43 → 19:34:25).
- **Status:** success.

### Step 2 — notebook (LUT)

- **Issue:** `jupyter nbconvert` not installed in `sal_new_vllm`. Installed
  with `pip install --quiet nbconvert nbclient`.
- **Issue:** Notebook visualization cell `c666c3f2` references prompt_len=256
  in `coarse_prompts`, but the 3B sweep starts at 320 → KeyError. Fixed the
  cell to filter the wish list against `winner_table.columns`.
- **Commit:** `e5955db` — "profile notebook: filter coarse_prompts against
  actual measured columns".
- **Run:** `jupyter nbconvert --to notebook --execute scripts/profile_results_analyze.ipynb --inplace`
  → success after fix.
- **Verify:** All four files exist under `data/Qwen/QWen2.5-3B-Instruct/`:
  - `chunk_setting_profile_results.csv` (61 KB)
  - `profile_results_best_method.csv` (947 B)
  - `profile_results_lut.json` (3.1 KB; grid 11 x 24, n_beams `[1..64]`,
    prompt_len `[320..4096]`)
  - `profile_results_lut.npz` (2.2 KB)
- **Status:** success.

## Step 3 — Best-of-n sweep on MATH-500

Sweeping n ∈ [2, 4, 8, 16, 32, 64]. Dataset lines in `recipes/best-of-n.yaml`
are already commented out, so Config defaults (`HuggingFaceH4/MATH-500`,
split `test`) take over.

### Driver

- **Action:** wrote `scripts/run_qwen3b_sweep.sh` (positional arg picks phase:
  `step3` MATH-500 bon, `step4` MATH-500 bs, `step5` AIME bon, `step6` AIME bs,
  or `all`). The driver edits the recipe yaml per-n, runs the trace then the
  replay, and writes per-step logs under `logs/qwen3b/`.
- **Commit:** `64acace` — "add qwen3b sweep driver (Steps 3-6 of QWEN3B_SWEEP.md)".
- **Incident:** First test invocation accidentally launched the driver in the
  foreground (with `head -2` piped on stdout). The driver had `HF_HUB_OFFLINE=1`
  carried over from the AIME driver; with that flag, vLLM's case-insensitive
  resolution of `Qwen/QWen2.5-3B-Instruct` failed (cached dir uses capital W
  while canonical id uses lowercase w). Killed by SIGPIPE before reaching the
  bug, but the log
  `logs/qwen3b/bon_math500_trace_n2.log` captured the diagnostic.
- **Fix:** removed `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, added
  explicit absolute `HF_HOME=$HF_HOME` and
  `TRANSFORMERS_CACHE=$HF_HOME`.

### Run (first attempt, killed)

- **Run:** `nohup bash scripts/run_qwen3b_sweep.sh step3 > logs/qwen3b_step3_master.log 2>&1 &`
  (PID 3823232). Started 19:38:34 on `n=2` trace.
- **Killed at 19:55:** user requested switching to the simpler MATH-500 prompt
  for MATH-500 phases and the long systematic-thinking prompt for AIME phases.
  No artifacts had been written yet — kill is clean.

### Prompt-preset refactor

- **Action:** `src/sal/config.py` now exposes both prompts as module-level
  constants (`_SYSTEM_PROMPT_MATH500`, `_SYSTEM_PROMPT_AIME`) and a sentinel
  `_SYSTEM_PROMPT_PRESET` (single-line) that picks the active one. The
  dataclass field `system_prompt: str = _ACTIVE_SYSTEM_PROMPT` references
  the module-level resolution.
- **Action:** `scripts/run_qwen3b_sweep.sh` now has a `set_system_prompt_preset`
  helper that flips `_SYSTEM_PROMPT_PRESET` via regex. Each phase calls it
  before iterating (MATH-500 phases set `MATH500`; AIME phases set `AIME`).
- **Commit:** `b37c9a8` — "config: expose system_prompt as a switchable
  preset (AIME | MATH500)".

### Run (second attempt)

- **Run:** `nohup bash scripts/run_qwen3b_sweep.sh step3 > logs/qwen3b_step3_master.log 2>&1 &`
  (PID 3900191). Master log: `logs/qwen3b_step3_master.log`; per-step logs:
  `logs/qwen3b/bon_math500_{trace,replay}_n${n}.log`.

### Step 3 progress (auto-recorded from master log)

| n  | trace start         | trace ok            | replay ok           | status |
|----|---------------------|---------------------|---------------------|--------|
| 2  | 2026-05-12 20:36:28 | 2026-05-12 21:27:02 | 2026-05-12 22:15:08 | ok     |
| 4  | 2026-05-12 22:15:08 | 2026-05-12 23:11:42 | 2026-05-13 00:04:50 | ok     |
| 8  | 2026-05-13 00:04:50 | 2026-05-13 01:12:04 | 2026-05-13 02:15:13 | ok     |
| 16 | 2026-05-13 02:15:13 | 2026-05-13 03:30:30 | 2026-05-13 04:41:10 | ok     |
| 32 | 2026-05-13 04:41:10 | 2026-05-13 06:14:02 | 2026-05-13 07:43:16 | ok     |
| 64 | 2026-05-13 07:43:16 | 2026-05-13 09:31:00 | (killed mid-replay) | **stopped** |

### Resume after Windows-side interruption (2026-05-13)

- **Context:** User reported their Windows machine auto-rebooted, interrupting
  the SSH session. The server-side processes were unaffected — the step3
  driver was launched with `nohup` and survived the disconnect. Verified at
  08:57 on 2026-05-13:
  - PID 3900191 still alive (etime 12h20m).
  - GPU at 94% utilization, 22.2 GiB used.
  - n=64 trace process (PID 366519) actively running.
- **Action:** wrote `scripts/run_qwen3b_sweep_chain.sh` — a wrapper that
  blocks on the step3 driver PID, verifies `PHASE_DONE step3` appears in the
  master log, then runs step4 → step5 → step6 sequentially via the existing
  driver. Each phase writes its own master log
  (`logs/qwen3b_step{4,5,6}_master.log`); chain status goes to
  `logs/qwen3b_chain_master.log`.
- **Run:** `nohup bash scripts/run_qwen3b_sweep_chain.sh 3900191 > /dev/null 2>&1 &`
  (chain PID 439877, started 2026-05-13 08:58:01). Currently blocked on
  `kill -0 3900191`; will fire step4 the moment step3 exits cleanly.
- **No commit yet for the chain script** — it's a one-off operational helper.
  Will commit at end-of-sweep along with this log update.

### Stop request (2026-05-13 ~10:11)

- **User asked to stop the work.** Killed in this order (all SIGTERM):
  chain PID 439877, driver PID 3900191, vLLM worker PID 481468, lingering
  vLLM child PID 481631. Verified no `run_qwen3b_sweep` / `best_of_n_task_trace`
  / `test_time_compute` / `beam_search` survivors. GPU returned to 0% / 49 MiB.
- **State at stop:** Step 3 — n=2..32 fully complete on disk; n=64 trace
  complete on disk; n=64 replay **incomplete** (killed mid-flight; partial
  output may exist under `data/Qwen/QWen2.5-3B-Instruct/MATH-500/` but should
  not be trusted). Steps 4, 5, 6 — never started.
- **Resume path (if needed later):** the per-`n` recipe state at stop time is
  whatever the driver last wrote — verify `recipes/best-of-n.yaml` settings
  before relaunch. Step 4 driver invocation: `bash scripts/run_qwen3b_sweep.sh step4`.
