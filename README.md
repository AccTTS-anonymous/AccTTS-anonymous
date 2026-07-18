# AccTTS

**AccTTS: Characterizing and Optimizing Workload Dynamics in Test-Time Scaling**

AccTTS is a computational optimization framework for test-time scaling (TTS). During each reasoning step, beams finish asynchronously, so the active beam count decreases while the contexts of surviving beams grow. This evolution causes small-$M$ GEMM inefficiency, reduces beam-level attention parallelism, and progressively shifts latency dominance toward attention.

AccTTS addresses these effects with two complementary adaptations:

- **Beam-adaptive GEMM execution** follows the changing beam dimension with profiled kernel designs.
- **Adaptive context parallelization** exploits growing contexts to recover attention parallelism while accounting for splitting overhead.

AccTTS uses offline profiling to identify efficient, hardware-dependent execution configurations. Integrated into vLLM, it achieves **1.08x to 1.43x end-to-end speedups** across GPUs, TTS algorithms, models, datasets, and compute budgets while preserving inference quality.

## Results

### Qwen-2.5-1.5B-Instruct on RTX 4090

<table>
  <tr>
    <td width="50%"><img src="figures/e2e_results_bon_MATH500_QWen.png" alt="Best-of-N on MATH-500 with Qwen-2.5-1.5B-Instruct"></td>
    <td width="50%"><img src="figures/e2e_results_bs_MATH500_QWen.png" alt="Beam search on MATH-500 with Qwen-2.5-1.5B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>MATH-500</b>, Best-of-N</td>
    <td align="center"><b>MATH-500</b>, Beam search</td>
  </tr>
  <tr>
    <td><img src="figures/e2e_results_bon_AIME_QWen.png" alt="Best-of-N on AIME with Qwen-2.5-1.5B-Instruct"></td>
    <td><img src="figures/e2e_results_bs_AIME_QWen.png" alt="Beam search on AIME with Qwen-2.5-1.5B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>AIME</b>, Best-of-N</td>
    <td align="center"><b>AIME</b>, Beam search</td>
  </tr>
</table>

### Qwen-2.5-7B-Instruct on A100

<table>
  <tr>
    <td width="50%"><img src="figures/e2e_results_bon_MATH500_Qwen7B.png" alt="Best-of-N on MATH-500 with Qwen-2.5-7B-Instruct"></td>
    <td width="50%"><img src="figures/e2e_results_bs_MATH500_Qwen7B.png" alt="Beam search on MATH-500 with Qwen-2.5-7B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>MATH-500</b>, Best-of-N</td>
    <td align="center"><b>MATH-500</b>, Beam search</td>
  </tr>
  <tr>
    <td><img src="figures/e2e_results_bon_AIME_Qwen7B.png" alt="Best-of-N on AIME with Qwen-2.5-7B-Instruct"></td>
    <td><img src="figures/e2e_results_bs_AIME_Qwen7B.png" alt="Beam search on AIME with Qwen-2.5-7B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>AIME</b>, Best-of-N</td>
    <td align="center"><b>AIME</b>, Beam search</td>
  </tr>
</table>

### Llama-3.2-1B-Instruct on RTX 4090

<table>
  <tr>
    <td width="50%"><img src="figures/e2e_results_bon_MATH500_Llama.png" alt="Best-of-N on MATH-500 with Llama-3.2-1B-Instruct"></td>
    <td width="50%"><img src="figures/e2e_results_bs_MATH500_Llama.png" alt="Beam search on MATH-500 with Llama-3.2-1B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>MATH-500</b>, Best-of-N</td>
    <td align="center"><b>MATH-500</b>, Beam search</td>
  </tr>
  <tr>
    <td><img src="figures/e2e_results_bon_AIME_Llama.png" alt="Best-of-N on AIME with Llama-3.2-1B-Instruct"></td>
    <td><img src="figures/e2e_results_bs_AIME_Llama.png" alt="Beam search on AIME with Llama-3.2-1B-Instruct"></td>
  </tr>
  <tr>
    <td align="center"><b>AIME</b>, Best-of-N</td>
    <td align="center"><b>AIME</b>, Beam search</td>
  </tr>
</table>

### Inference Quality

![Best-of-N accuracy on MATH-500](figures/accuracy_verify_aime_llama1b_multi_round.png)

Each result reports the mean accuracy over five independent runs, with error bars showing the standard error of the mean (SEM).

### GEMM Efficiency

![GEMM efficiency across beam counts](figures/gemm_efficiency_profile.png)

### Attention Latency

![Attention latency comparison](figures/vllm_heuristic_profiling_compare_attn_latency.png)

AccTTS profiles KV-cache-block-aligned chunk sizes for each beam-count and context-length pair and selects the lowest-latency configuration. The irregular choices below show why a fixed heuristic cannot consistently identify the best configuration.

<p align="center">
  <img src="figures/chunk_profile_results_visualize_Qwen.png" width="49%" alt="Profiled chunk configurations for Qwen-2.5-1.5B-Instruct">
  <img src="figures/chunk_profile_results_visualize_Llama.png" width="49%" alt="Profiled chunk configurations for Llama-3.2-1B-Instruct">
</p>

### Runtime Adaptation

![Latency and attention parallelism over decoding](figures/latency_breakdown_timeline.png)

AccTTS adapts attention parallelism as the beam count decreases and the context length grows during decoding.

### Optimization Attribution

![Contributions of GEMM and attention optimizations](figures/attribution_analysis.png)

The attribution study separates the speedups contributed by beam-adaptive GEMM execution and adaptive context parallelization.

## Repository Structure

- `.github/`: repository workflows.
- `agent_markdown/`: experiment notes and sweep records.
- `assets/`: repository assets.
- `figures/`: paper and evaluation figures.
- `nvidia_toolkit_data/`: processed profiling results and analysis notebooks.
- `recipes/`: model and TTS configuration files.
- `scripts/`: profiling, tracing, replay, evaluation, and plotting scripts.
- `src/`: TTS pipeline and AccTTS runtime implementation.
- `tests/`: tests and basic validation utilities.

Large generated outputs, local model caches, raw experiment data, and binary Nsight reports are not maintained in this repository.

## Installation

```bash
conda create -n acctts python=3.11
conda activate acctts
pip install -e .
```

A CUDA-capable PyTorch, vLLM, and Triton environment is required. Some models also require authentication through the Hugging Face CLI.

## Quick Start

Run Best-of-N:

```bash
python scripts/test_time_compute.py recipes/best-of-n.yaml
```

Run beam search:

```bash
python scripts/test_time_compute.py recipes/beam-search.yaml
```

Model paths, datasets, compute budgets, PRM settings, and AccTTS optimizations are configured in the corresponding YAML files under `recipes/`.

## Profiling and Evaluation

The main profiling and evaluation entry points are located under `scripts/`:

- `gemm_best_templates_collect.py` profiles GEMM kernel designs.
- `chunk_setting_profile.py` profiles context-splitting configurations.
- `chunk_setting_LUT_builder.py` prepares runtime attention configurations.
- `test_time_compute.py` runs end-to-end TTS evaluation.
- `e2e_results_analyze.ipynb` and `attribution_analysis.ipynb` generate performance results.

Processed profiling outputs used by the analysis scripts are provided under `nvidia_toolkit_data/`.

## Acknowledgment

This codebase builds on [Hugging Face Search and Learn](https://github.com/huggingface/search-and-learn) and extends it with workload tracing, GEMM profiling, adaptive context parallelization, and runtime integration for AccTTS.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
