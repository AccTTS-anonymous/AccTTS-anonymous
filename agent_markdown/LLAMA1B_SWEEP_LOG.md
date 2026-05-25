# Llama-3.2-1B-Instruct Sweep Log

Run started: 2026-05-13

This log records every action taken while executing LLAMA1B_SWEEP.md.

---

## Pre-flight (2026-05-13 ~10:30)

- **GPU:** idle (0% util, 49 MiB residual) — verified clean after Qwen-3B
  sweep stop.
- **Gemm ladder** (`scripts/gemm_best_templates_collect.py:24`): already
  `[1, 2, 4] + list(range(8, 65, 8))` from the Qwen-3B sweep. **No edit/commit
  needed for Step 1.**
- **Recipes:** both already declare `model_path: meta-llama/Llama-3.2-1B-Instruct`
  in the working tree. Other knobs (n, gemm_opt, chunk_size, dataset) get
  rewritten per-iteration by the driver, so the current values don't matter.
- **Step 2 LUT:** all four files present under
  `data/meta-llama/Llama-3.2-1B-Instruct/`:
  - `chunk_setting_profile_results.csv` (64 KB)
  - `profile_results_best_method.csv` (1.5 KB)
  - `profile_results_lut.json` (3.4 KB) ← runtime loads this when `chunk_size: dynamic`
  - `profile_results_lut.npz` (2.3 KB)
  Step 2 explicitly skipped per LLAMA1B_SWEEP.md.
- **HF token:** `meta-llama/Llama-3.2-1B-Instruct` is gated. The token is
  exported in the launch environment, NOT hardcoded into the driver script
  (driver guards on `${HF_TOKEN:?…}` so it fails fast if missing). Token is
  **not committed**.

## Step 1 — GEMM templates for Llama-3.2-1B

- **Run:** `python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml`
  (started 18:48, finished 18:55 — 7 min). Log: `logs/llama1b_gemm_collect.log`.
- **Stdout:** "Pre-compile finished for 11 shapes: 1..64. Cache populated under
  ~/.cache/vllm/torch_compile_cache/."
- **Cache:** `~/.cache/vllm/torch_compile_cache/7334850221/` (different hash
  than Qwen-3B's `c473fd9511`, as expected since the parent hash is keyed on
  model + INDUCTOR_COMPILE_CONFIG).
- **Status:** success. Exit 0.

## Step 2 — Chunk-setting profile + LUT

Skipped per instructions; all four LUT files were already present in pre-flight.

## Driver

- **Action:** wrote `scripts/run_llama1b_sweep.sh` — mirror of the Qwen-3B
  driver, with two differences:
  1. `: "${HF_TOKEN:?…}"` guard at the top (Llama-3.2 is gated; fail fast if
     the launcher forgot to export the token).
  2. Per-step logs land under `logs/llama1b/` (not `logs/qwen3b/`).
  Same per-`n` recipe-edit logic, same `NS=(2 4 8 16 32 64)` grid, same
  `beam_width: 2` rule for `n=2` in the beam-search phase.

## Step 3 — Best-of-n sweep on MATH-500

- **Launch (2026-05-13 18:55:35):**
  `nohup bash scripts/run_llama1b_sweep.sh step3 > logs/llama1b_step3_master.log 2>&1 &`
  Driver PID 1373367. Per-step logs: `logs/llama1b/bon_math500_{trace,replay}_n${n}.log`.
- **Chain (2026-05-13 18:55:40):**
  `nohup bash scripts/run_llama1b_sweep_chain.sh 1373367 > /dev/null 2>&1 &`
  Chain PID 1373636. Will fire step4 → step5 → step6 once driver exits clean.
- **n=2 trace startup verified:** model `meta-llama/Llama-3.2-1B-Instruct`
  loaded in 0.7s (2.3 GiB), torch.compile cache hit on `bcea9e84ea` (the vLLM
  default-config cache from prior Llama profile work). No HF auth errors.

### Step 3 progress (from master log)

| n  | trace start         | trace ok            | replay ok           |
|----|---------------------|---------------------|---------------------|
| 2  | 2026-05-13 18:55:35 | 2026-05-13 19:25:02 | 2026-05-13 19:51:07 |
| 4  | 2026-05-13 19:51:07 | 2026-05-13 20:28:26 | 2026-05-13 21:00:36 |
| 8  | 2026-05-13 21:00:36 | 2026-05-13 21:47:51 | 2026-05-13 22:27:36 |
| 16 | 2026-05-13 22:27:36 | 2026-05-13 23:30:43 | 2026-05-14 00:22:18 |
| 32 | 2026-05-14 00:22:18 | 2026-05-14 01:39:59 | 2026-05-14 02:43:58 |
| 64 | 2026-05-14 02:43:58 | 2026-05-14 04:21:54 | 2026-05-14 05:43:44 |

**Step 3 PHASE_DONE at 2026-05-14 05:43:44.** Total wall time: 10h48m.
Chain fired step 4 at 05:44:41.

## Step 4 — Beam-search sweep on MATH-500

- Launched by chain at 2026-05-14 05:44:41. Master log: `logs/llama1b_step4_master.log`.
- Per-step logs: `logs/llama1b/bs_math500_{trace,replay}_n${n}.log`.

### Step 4 progress before stop

| n  | trace start         | trace ok            | replay ok           |
|----|---------------------|---------------------|---------------------|
| 2  | 2026-05-14 05:44:41 | 2026-05-14 05:57:39 | 2026-05-14 06:08:33 |
| 4  | 2026-05-14 06:08:33 | 2026-05-14 06:30:18 | 2026-05-14 06:46:22 |
| 8  | 2026-05-14 06:46:22 | 2026-05-14 07:35:52 | 2026-05-14 07:59:38 |
| 16 | 2026-05-14 07:59:38 | 2026-05-14 09:32:36 | (killed mid-replay) |
| 32 | —                   | —                   | —                   |
| 64 | —                   | —                   | —                   |

### Stop request (first attempt)

- **User asked to stop.** Killed (all SIGTERM): chain PID 1373636,
  step4 driver PID 2302820, beam_search_beam workers PIDs 2611207, 2611412.
  GPU returned to 0% / 49 MiB.
- **Reason for stop (per user):** the MATH-500 beam-search results from this
  first attempt had problems. User then updated SamplingParams in
  `scripts/beam_search_beam.py:290` — replaced
  `max_tokens=min(256, remaining)` with `max_tokens=remaining`, removing the
  256-token cap on per-iteration continuation steps. This change is in the
  working tree, uncommitted at relaunch time.

### Step 4 relaunch (2026-05-14 10:09:14, full redo with the fix)

- **User asked to redo Step 4 with the new SamplingParams; existing MATH-500
  beam-search results may be overwritten.** Also greenlit chaining steps 5
  and 6 (AIME bon + AIME bs) after Step 4 completes.
- **Driver script:** `scripts/run_llama1b_sweep_chain456.sh` — sequential
  invoker that runs `step4 → step5 → step6` via the existing
  `scripts/run_llama1b_sweep.sh`. Status: `logs/llama1b_chain456_master.log`.
- **Launch:** `nohup bash scripts/run_llama1b_sweep_chain456.sh > /dev/null 2>&1 &`
  (chain PID 2669950). HF_TOKEN exported in the launch env (guard in driver +
  chain catches missing token).
- **n=2 trace startup verified at 10:09:** Llama loaded in 0.65s, torch.compile
  cache hit on `bcea9e84ea`, no auth errors.

### Step 4 redo progress (from master log)

| n  | trace start         | trace ok            | replay ok           |
|----|---------------------|---------------------|---------------------|
| 2  | 2026-05-14 10:09:14 | 2026-05-14 10:37:31 | 2026-05-14 11:00:33 |
| 4  | 2026-05-14 11:00:33 | 2026-05-14 11:45:34 | 2026-05-14 12:14:17 |
| 8  | 2026-05-14 12:14:17 | 2026-05-14 13:55:58 | 2026-05-14 14:36:07 |
| 16 | 2026-05-14 14:36:07 | 2026-05-14 17:36:36 | 2026-05-14 18:29:39 |
| 32 | 2026-05-14 18:29:39 | 2026-05-14 22:52:07 | 2026-05-15 00:02:45 |
| 64 | 2026-05-15 00:02:45 | 2026-05-15 06:28:25 | 2026-05-15 08:02:09 |

**Step 4 PHASE_DONE at 2026-05-15 08:02:09.** Total wall time: ~21h53m.
The fixed SamplingParams (`max_tokens=remaining`) make beams generate fuller
continuations — roughly 1.7-2.3x slower per `n` than the capped version.
Chain fired step 5 at 08:02:09.

## Step 5 — Best-of-n sweep on AIME

- Launched by chain at 2026-05-15 08:02:09. Master log: `logs/llama1b_step5_master.log`.

| n  | trace start         | trace ok            | replay ok           |
|----|---------------------|---------------------|---------------------|
| 2  | 2026-05-15 08:02:09 | 2026-05-15 08:14:58 | 2026-05-15 08:25:30 |
| 4  | 2026-05-15 08:25:30 | 2026-05-15 08:41:01 | 2026-05-15 08:53:41 |
| 8  | 2026-05-15 08:53:41 | 2026-05-15 09:16:32 | 2026-05-15 09:33:52 |
| 16 | 2026-05-15 09:33:52 | 2026-05-15 10:02:22 | 2026-05-15 10:23:53 |
| 32 | 2026-05-15 10:23:53 | 2026-05-15 10:56:41 | 2026-05-15 11:22:02 |
| 64 | 2026-05-15 11:22:02 | 2026-05-15 11:59:56 | 2026-05-15 12:30:42 |

**Step 5 PHASE_DONE at 2026-05-15 12:30:42.** Total wall time: ~4h28m.
Chain fired step 6 at 12:30:42.

## Step 6 — Beam-search sweep on AIME

- Launched by chain at 2026-05-15 12:30:42. Master log: `logs/llama1b_step6_master.log`.
- Final phase of the sweep (AIME beam-search, uses the fixed
  `beam_search_beam.py`).

### Step 6 progress (from master log)

| n  | trace start         | trace ok            | replay ok           |
|----|---------------------|---------------------|---------------------|
| 2  | 2026-05-15 12:30:42 | 2026-05-15 12:42:09 | 2026-05-15 12:51:34 |
| 4  | 2026-05-15 12:51:34 | 2026-05-15 13:14:53 | 2026-05-15 13:30:08 |
| 8  | 2026-05-15 13:30:08 | 2026-05-15 14:15:49 | 2026-05-15 14:35:28 |
| 16 | 2026-05-15 14:35:28 | 2026-05-15 15:56:16 | 2026-05-15 16:22:07 |
| 32 | 2026-05-15 16:22:07 | 2026-05-15 18:20:29 | 2026-05-15 18:57:33 |
| 64 | 2026-05-15 18:57:33 | 2026-05-15 21:41:34 | 2026-05-15 22:29:34 |

**Step 6 PHASE_DONE at 2026-05-15 22:29:34.** Total wall time: ~9h59m.
Chain logged `CHAIN456_COMPLETE all phases done` at 22:29:34.

- **Final verification 2026-05-15 (post-completion):** all chain/driver/python
  processes exited cleanly. AIME beam-search outputs for n ∈ {2,4,8,16,32,64}
  all present on disk — `beam_search_n{n}`, `.csv`, `_timing.csv`,
  `beam_search_request_timings_{vllm,ours}_n{n}.csv`,
  `beam_search_per_step_ours_n{n}.csv`. No OOMs, no skipped problems.

## LLAMA1B_SWEEP COMPLETE

All six steps done. `logs/llama1b_step6_master.log` ends with
`LLAMA1B_SWEEP_COMPLETE` at 2026-05-15 22:29:34; chain master log ends with
`CHAIN456_COMPLETE all phases done`.

| Step | Phase                       | Done                |
|------|-----------------------------|---------------------|
| 1-2  | gemm cache / chunk LUT      | (prereqs, pre-done) |
| 3    | best-of-n / MATH-500        | (earlier)           |
| 4    | beam-search / MATH-500      | 2026-05-15 08:02:09 |
| 5    | best-of-n / AIME            | 2026-05-15 12:30:42 |
| 6    | beam-search / AIME          | 2026-05-15 22:29:34 |
