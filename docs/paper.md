# InferLite: hardware-aware LLM inference configuration without invented scores

## Abstract

Choosing quantization, context length, and decode length for LLM inference is a multi-objective problem (latency, throughput, memory, energy, quality). Most published tables assume CUDA kernels that a laptop or free Colab GPU may not have. InferLite probes the machine, times only what loads, and labels the rest **unsupported**. On Apple MPS, GPT-2 FP16 is faster than FP32 (211 vs 146 tok/s in the measurement suite). On a Colab Tesla T4, TinyLlama Hugging Face FP16 is faster than bitsandbytes INT4, while INT4 uses less GPU memory. The same T4 later timed TinyLlama **GGUF Q4_K_M via llama.cpp** at 172.5 tok/s (P95 106 ms) after installing a CUDA 12.4 wheel — a different backend, not mixed into the bitsandbytes Pareto.

The optimizer question is whether a hardware-aware multi-objective search can recover a near-Pareto front with far fewer measurements than exhaustive grid search. On a **40-configuration** GPT-2 / MPS space, InferLite reaches 0.86× of grid hypervolume at budget 4 (mean over five seeds; 95% CI [0.83, 0.90]) versus random 0.61×. At budget 16 both methods are ~0.97× and random is slightly ahead. On a **30-configuration** TinyLlama / T4 space with `warmup_runs=1`, the ranking flips at small budgets: random 0.96× vs InferLite 0.49× at budget 4; they meet near 0.97× at budget 8. An 8-point single-seed MPS pilot had random 0.32 vs InferLite 0.30. A 4-point T4 pilot is kept as a negative control: leave-one-out P95/tokens/s R² are −6.66 / −8.44 because n=4 is too small; the n=30 T4 predictor is usable (P95 R² 0.84, tokens/s 0.68). Energy is unsupported without NVML. No simulated tokens/s are reported as measurements.

## Research question

**Can a hardware-aware multi-objective optimizer identify near-Pareto-optimal LLM inference configurations using substantially fewer measurements than exhaustive search?**

Sub-questions: (1) What can actually run on a MacBook MPS and a free T4? (2) What is the measured Pareto front among those configs? (3) Under what search-space sizes and evaluation budgets does InferLite outperform random search? (4) Which features (hardware, quantization, workload) the surrogate actually uses?

## Related work

vLLM, TensorRT-LLM, AWQ, GPTQ, bitsandbytes, and llama.cpp provide kernels and serving stacks. Auto-tuning papers often assume those stacks are installed. InferLite is complementary: a measurement protocol and search loop that **refuses to score missing kernels**. Queueing simulation remains in the repo for capacity planning and is labeled as simulation, not a model benchmark.

## Methodology

**Real vs simulation.** Wall-clock generation goes through `research.engine.run_benchmark`. Missing libraries return `status=unsupported` with null metrics. `python -m research simulate` is a Poisson queueing model and prints a disclaimer.

**Metrics.** Greedy decode (`temperature=0`), warmup discarded. TTFT, inter-token latency, tokens/s, P50/P95/P99, mean, std, 95% CI (normal approximation on repeated runs; Student-t across optimizer seeds), RSS (MPS) or CUDA allocated bytes (T4), load time. Energy: NVML if present; otherwise unsupported. Quality: optional short-passage perplexity; MMLU/GSM8K/HumanEval are not fabricated.

**Search.** Capability filter, then four strategies on the **same** space: grid, random (budget B), hardware heuristic (one config), InferLite (diverse seed + ridge surrogate, budget B). Objectives: minimize P95 and memory, maximize tokens/s. Hypervolume is 2-D (memory vs throughput) relative to the grid front. The scale protocol times the grid once and **replays** budgeted strategies from that cache, so InferLite’s sequential picks still use measured values. InferLite only fits the ridge model after three measured points, so budget 2 is a diverse/heuristic seed rather than surrogate search.

**Seeds and budgets.** Optimizer seeds `{42, 123, 456, 789, 1000}`. Budgets `{2, 4, 8, 16}`. Report mean, sample std, and 95% t-interval of hypervolume / grid. The key comparison is InferLite vs random, not InferLite vs a marketing baseline.

**Predictor.** Independent ridge models for P95, tokens/s, and memory. Features: method one-hot, CUDA/MPS/GPU memory, log context / new tokens / batch. Ablations drop hardware, quantization, or workload groups. Validation: leave-one-out MAE, RMSE, R². Negative R² is reported, not patched.

**Reproducibility.** YAML configs, `python -m research suite|optimize`, JSON/CSV + figures. Dependencies in `backend/requirements.txt` (Mac) and `backend/requirements-colab.txt` (Colab; do not reinstall torch).

## Architecture

```
configs/*.yaml → research.runner → engine / search / predictor → JSON/CSV + figures
```

Capability probe → load or unsupported → warmup → timed runs → Pareto over **measured** rows only.

## Experiments

| Study | Hardware | Model | What was timed |
|-------|----------|-------|----------------|
| Measurement suite | MacBook MPS | GPT-2 / DistilGPT-2 | Precision, KV, speculative, batching |
| T4 lite | Colab Tesla T4 | TinyLlama 1.1B | FP16, INT8/INT4 (bnb); AWQ unsupported in that suite |
| T4 llama.cpp | Colab Tesla T4 | TinyLlama 1.1B Q4_K_M | GGUF, `n_gpu_layers=-1`, 172.5 tok/s |
| Search scale | MacBook MPS | GPT-2 | **40-config** space; 5 seeds; budgets 2/4/8/16 |
| Search scale | Colab Tesla T4 | TinyLlama 1.1B | **30-config** space; 5 seeds; budgets 2/4/8/16 |
| Search pilot | MacBook MPS | GPT-2 | 8-config space; seed 42; budget 4 |
| Search pilot | Colab Tesla T4 | TinyLlama 1.1B | 4-config space; InferLite 0.85× vs random 0.041× |

Llama-3-8B, Mistral, and Qwen were **not** timed here. TinyLlama is the Llama-family model that fits a free T4. Cross-device transfer (train on MPS, test on T4) is future work.

Workloads in the scale study: context `{32, 48, 64, 96, 128}`, new tokens `{8, 12, 16, 24}`, batch 1, methods fp32 and fp16 (the methods this Mac can load).

## Results

### Measurement suites

See the root README and `docs/results/macbook_mps_gpt2/`, `docs/results/colab_t4_lite/`, `docs/results/colab_t4_gguf/`. Headline: MPS FP16 beats FP32 on GPT-2 throughput; T4 Hugging Face FP16 beats bitsandbytes INT8/INT4 on TinyLlama throughput; T4 INT4 wins HF memory; T4 llama.cpp Q4_K_M is a separate 172.5 tok/s measurement.

### Search scale (this paper’s main table)

40 wall-clock GPT-2 configs on MPS, 2026-08-25 19:41 UTC. Grid measured once. Random / InferLite / heuristic replay those records. Five seeds, budgets 2 / 4 / 8 / 16.

| Budget | InferLite HV/grid mean (std) [95% CI] | Random HV/grid mean (std) [95% CI] |
|-------:|--------------------------------------:|-----------------------------------:|
| 2 | 0.48 (0.24) [0.18, 0.78] | 0.31 (0.36) [−0.14, 0.76] |
| 4 | 0.86 (0.03) [0.83, 0.90] | 0.61 (0.39) [0.12, 1.10] |
| 8 | 0.92 (0.02) [0.90, 0.95] | 0.80 (0.38) [0.33, 1.27] |
| 16 | 0.97 (0.03) [0.93, 1.01] | 0.97 (0.01) [0.95, 0.99] |

At budget 4 InferLite recovers 0.86× of exhaustive hypervolume using 10% of the measurements, with a tight interval. Random’s mean is lower and its interval is wide: seed-to-seed HV at B=4 ranges from 0.11 to 0.95. At budget 16 (~40% of the space) random is slightly ahead (0.973 vs 0.971). The heuristic stays at 0.18×. InferLite is therefore sample-efficient and stable on this space; it is **not** uniformly better than random once the budget is large.

![Hypervolume vs budget](results/optimizer_macbook_scale/figures/hv_vs_budget.png)

### Search scale: 30-point T4 TinyLlama

30 wall-clock TinyLlama configs on a Colab Tesla T4, 2026-08-25 20:08 UTC. `warmup_runs=1`. Same seeds and budgets.

| Budget | InferLite HV/grid mean (std) [95% CI] | Random HV/grid mean (std) [95% CI] |
|-------:|--------------------------------------:|-----------------------------------:|
| 2 | 0.33 (0.31) [−0.05, 0.71] | 0.61 (0.45) [0.05, 1.17] |
| 4 | 0.49 (0.40) [−0.01, 0.99] | 0.96 (0.02) [0.93, 0.98] |
| 8 | 0.97 (0.03) [0.93, 1.01] | 0.97 (0.02) [0.94, 0.99] |
| 16 | 0.995 (0.005) [0.988, 1.001] | 0.990 (0.011) [0.976, 1.004] |

Random is ahead at budgets 2 and 4. InferLite catches up at 8–16. Heuristic is 0.19×. Do not copy the Mac budget-4 ranking onto this table.

![T4 hypervolume vs budget](results/optimizer_colab_t4_scale/figures/hv_vs_budget.png)

### Pilot: 8-point MPS, seed 42, budget 4

Kept as a negative control for overclaiming. InferLite 0.30× vs random 0.32× vs heuristic 0.22× vs grid 1.00. On this seed random slightly outperformed InferLite. n=8 is too small to rank optimizers.

![Search vs baselines (n=8)](results/optimizer_macbook/figures/search_vs_baselines.png)

### Pilot: 4-point T4 TinyLlama, budget 2

Same protocol on a Colab Tesla T4. Grid: fp16/INT4 × context 32/64, 8 new tokens. InferLite 0.85× grid HV; random 0.041×; heuristic 0.22×. The first FP16-c32 generate had `warmup_runs=0` (6.1 tok/s vs 31.8 tok/s on the later FP16-c64 job). Leave-one-out P95/tokens/s R² are negative at n=4; memory R² is 0.999. The predictor fails to generalize with only four T4 observations. The n=30 scale study is the T4 predictor to cite.

![T4 search vs baselines](results/optimizer_colab_t4/figures/search_vs_baselines.png)

### Ablation (leave-one-out ridge on the 40-point MPS grid)

| Variant | Memory R² | Tokens/s R² | P95 R² |
|---------|----------:|------------:|-------:|
| Full | 0.999 | 0.61 | 0.71 |
| No hardware | 0.999 | 0.61 | 0.71 |
| No quantization | −0.17 | −0.17 | 0.59 |
| No workload | 0.998 | 0.64 | 0.02 |

Quantization features carry memory and throughput (FP16 vs FP32). Workload features carry P95 (context and new tokens). Hardware features are constant on one Mac, so dropping them changes nothing.

![Predictor ablation](results/optimizer_macbook_scale/figures/predictor_ablation.png)

### Ablation (leave-one-out ridge on the 30-point T4 grid)

| Variant | Memory R² | Tokens/s R² | P95 R² |
|---------|----------:|------------:|-------:|
| Full | 0.9999 | 0.68 | 0.84 |
| No hardware | 0.9999 | 0.68 | 0.84 |
| No quantization | −0.24 | −0.22 | 0.48 |
| No workload | 0.9998 | 0.71 | 0.21 |

Same feature story as MPS: quantization carries memory and throughput; workload carries P95. Hardware features are constant on one T4.

![T4 predictor ablation](results/optimizer_colab_t4_scale/figures/predictor_ablation.png)

## Limitations

The 40-point MPS study is still GPT-2, not a 7B serving stack. On T4 n=30, random beats InferLite at budgets 2 and 4; InferLite only matches it at 8–16. Random’s 95% t-interval at small budgets can include values below 0 because n_seeds=5 and HV is unstable; HV/grid cannot be negative. InferLite vs random ranking also flipped between the n=8 single-seed MPS pilot and the n=40 five-seed mean. The T4 first FP16 row in the n=4 pilot is a cold start (`warmup_runs=0`); the n=30 run used warmup. MPS RSS is not CUDA `max_memory_allocated`. llama.cpp’s 9.125 MB engine snapshot is not llama.cpp VRAM. Speculative decoding and continuous batching in the measurement suite are research loops, not vLLM. Sliding-window KV is prompt truncation. Energy, MMLU, Mac→T4 predictor transfer, and Llama/Mistral/Qwen 7B+ are unsupported here. The Colab GPTQ bar from an earlier session loaded dense TinyLlama and is not cited as GPTQ.

## Conclusion

InferLite’s contribution is an honest measurement-and-search loop: probe, time, Pareto, compare search strategies across budgets and seeds, ablate the surrogate, and refuse fake kernels. On the 40-point MPS space, InferLite identified a near-grid Pareto front at 4 measurements with low seed variance; random caught up at 16. On the 30-point T4 space, random was ahead at 4 measurements and InferLite only caught up at 8–16. On the 8-point single-seed MPS pilot, random slightly beat InferLite. On four T4 points the predictor’s P95 and tokens/s R² are negative; on thirty T4 points they are 0.84 and 0.68. Those statements can sit in the same paper because none of the numbers were invented.

## Reproducibility commands

```bash
cd backend
source .venv/bin/activate
python -m research capabilities
python -m research suite --config ../configs/macbook_cpu.yaml
python -m research optimize --config ../configs/optimizer_macbook_scale.yaml
python -m research optimize --config ../configs/optimizer_macbook.yaml
python -m research predict --bundle results/optimizer_macbook_scale/optimizer_macbook_scale.json
python -m research optimize --config ../configs/optimizer_colab_t4_scale.yaml
python -m pytest
```

Colab: `notebooks/inferlite_colab.ipynb`. Install only `requirements-colab.txt`. T4 scale artifacts: `docs/results/optimizer_colab_t4_scale/`.
