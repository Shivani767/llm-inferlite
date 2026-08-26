# InferLite: Hardware-Aware Multi-Objective LLM Inference Optimization

**Shivani Bhandari**  
Software: [https://github.com/Shivani767/llm-inferlite](https://github.com/Shivani767/llm-inferlite)  
License: Apache 2.0

---

## Abstract

Selecting quantization, context length, and decode length for large language model (LLM) inference is a multi-objective problem: latency, throughput, and memory compete, and the feasible set depends on which kernels are actually installed. Published serving tables often assume CUDA stacks (vLLM, AWQ, GPTQ, TensorRT-LLM) that a laptop or free cloud GPU may not have. InferLite is a measurement-and-search framework that **probes the machine, times only configurations that load, and labels the rest `unsupported` with null metrics**. It then compares exhaustive **grid search**, **random search**, a **hardware heuristic**, and **InferLite** (a ridge surrogate with Pareto-and-diversity acquisition) on the **same** capability-filtered space.

The research question is whether a hardware-aware multi-objective optimizer can identify near-Pareto-optimal LLM inference configurations **under a limited evaluation budget**. Whether that budget is *substantially* smaller than exhaustive search is an empirical outcome, not a premise. We compare grid, random, a hardware heuristic, and InferLite on two measured scale studies. On Apple MPS, a 40-configuration GPT-2 space (fp32/fp16 × five contexts × four decode lengths) is timed once; budgeted strategies **replay** those records across five seeds and budgets $\{2,4,8,16\}$. At budget 4 ($4/40=10\%$ of the grid) InferLite achieved a higher mean hypervolume than random: $0.86\times$ of the exhaustive front (95% Student-$t$ CI $[0.83, 0.90]$) versus $0.61\times$ (CI $[0.12, 1.10]$). At budget 16 both methods are $\approx 0.97\times$ and random is slightly ahead ($0.973$ vs $0.971$). On a Colab Tesla T4, a 30-configuration TinyLlama 1.1B space (fp16/INT4 × five contexts × three decode lengths, `warmup_runs=1`) shows the **opposite** ranking at the same small budgets: random $0.96\times$ vs InferLite $0.49\times$ at budget 4. InferLite only matches random at budgets 8–16. The effectiveness of surrogate-guided inference optimization is therefore **hardware- and budget-dependent**. InferLite does not always beat random.

A leave-one-out ridge predictor fits memory almost perfectly when quantization is observed ($R^2 \approx 0.999$ on both scale grids). P95 and tokens/s are usable at $n=40$ MPS ($R^2$ $0.71$ / $0.61$) and $n=30$ T4 ($0.84$ / $0.68$), and collapse when the relevant feature group is dropped. A four-point T4 pilot yields negative P95/tokens/s $R^2$ ($-6.66$ / $-8.44$); those numbers are retained as a dataset limit, not patched. Energy is **not** an optimizer objective: the Mac scale study has no NVML (`energy.supported=false`); the T4 scale study records an NVML instant-watt probe but per-job `energy_j` is null. Llama-3-8B, Mistral, Qwen 7B+, vLLM, TensorRT-LLM, and old playground clocks (AWQ 71 tok/s, GPTQ 79 tok/s, TensorRT-LLM 295 tok/s, $4.34\times$ speculative) are **not** measurements in this paper.

---

## 1. Introduction

Deploying an LLM locally is not only a modeling problem. The operator must choose a precision or quantization method, a context length, and a decode length under latency, throughput, and memory constraints. Those choices interact with hardware: bitsandbytes NF4 requires CUDA [4, 5]; AWQ and GPTQ typically require CUDA plus specialized loaders [2, 3]; vLLM’s PagedAttention is a CUDA serving kernel [1]; Apple Metal Performance Shaders (MPS) exposes a different memory API (process RSS, not `torch.cuda.max_memory_allocated`). A configuration table that silently fills in missing kernels is not a measurement.

This paper describes **InferLite**, an open-source research loop that treats missing kernels as first-class outcomes. Every experiment record has `status ∈ {measured, unsupported, error}`. Unsupported and error rows have **null metrics**. Search strategies never rank a method last as a proxy for “does not run.”

On top of that protocol we ask a systems question that is standard in auto-tuning but rarely stated honestly for *local* LLM inference: **can a cheap surrogate recover a near-exhaustive Pareto front under a limited measurement budget?** Random search is a strong baseline when few dimensions matter [6]. Sequential model-based optimization can help when evaluations are expensive [7, 8], but a surrogate trained on four noisy points can be worse than sampling. We therefore compare InferLite to grid, random, and a one-shot hardware heuristic **on the same measured points**, at multiple budgets, with multiple seeds. The claim is not that InferLite always beats random. The result of interest is **when** surrogate search helps — and on these artifacts that answer is hardware- and budget-dependent.

Two hardware environments are used because they are what this project actually has:

- a MacBook with Apple MPS, timing Hugging Face `gpt2` (and DistilGPT-2 as a speculative draft);
- a Google Colab Tesla T4, timing TinyLlama 1.1B Chat [9] under Hugging Face Transformers (fp16, bitsandbytes INT8/INT4) and, in a separate study, llama.cpp GGUF Q4_K_M.

We do not stack those tables. We do not cite simulated Llama-3 / TensorRT numbers from an earlier playground. The contribution is the protocol and the measured comparison, including negative results.

---

## 2. Research Question

**Can a hardware-aware multi-objective optimizer identify near-Pareto-optimal LLM inference configurations under a limited evaluation budget?**

That question is deliberately **neutral**. Whether InferLite uses *substantially* fewer measurements than exhaustive search, and whether it beats random at a given budget, are results to be measured, not assumptions in the headline.

Operationally, “near-Pareto-optimal” means high **throughput–memory hypervolume relative to the exhaustive grid** on the capability-filtered space. “Limited evaluation budget” means $B \in \{2,4,8,16\}$ versus grid size $N=40$ (MPS) or $N=30$ (T4).

Sub-questions, all answered from artifacts in `docs/results/`:

1. What methods actually load on this MacBook MPS and this Colab T4?
2. What is the measured Pareto front among those methods (measurement suites, not the optimizer)?
3. Under which search-space sizes, hardware, and budgets does InferLite achieve a **higher mean** hypervolume than random search?
4. Which feature groups (hardware, quantization, workload) a ridge predictor uses, as measured by leave-one-out $R^2$?

The research question is **when** InferLite beats random, not a claim that it always does. The $n=8$ MPS pilot (seed 42) had random $0.32\times$ vs InferLite $0.30\times$. The $n=30$ T4 scale study has random ahead at $B=2$ and $B=4$. Those outcomes are part of the result. Published `budget_sweep.json` files still store an earlier wording of the question (“substantially fewer measurements”); the artifacts and numbers are unchanged.

---

## 3. Contributions

1. **An honest measurement protocol** for local LLM inference: capability probe, wall-clock greedy decode, explicit `unsupported` / `error` statuses, and a prohibition on inventing tokens/s, watts, or quality scores (`backend/research/engine.py`, `schema.py`).
2. **A capability-filtered search space** over method, context, decode length, and batch, with unsupported methods dropped before ranking (`search_space.py`).
3. **Four search strategies on the same space**: exhaustive grid, uniform random of size $B$, a one-evaluation hardware heuristic, and InferLite (diverse seed, then ridge surrogate with predicted-Pareto + diversity acquisition) (`optimizer.py`).
4. **A replay scale protocol**: time the grid once; replay budgeted strategies from the cache so hypervolume vs budget is not confounded by reloading the model (`run_budget_sweep`). Five seeds $\{42,123,456,789,1000\}$; Student-$t$ 95% intervals (`metrics.mean_std_ci95`).
5. **A ridge performance predictor** with leave-one-out validation and feature-group ablation (hardware / quantization / workload), including negative $R^2$ when $n=4$ on T4 (`predictor.py`).
6. **Seven measured study artifacts** across two hardware environments (Apple MPS and Colab Tesla T4), with figures: two scale-search studies (`optimizer_macbook_scale/`, `optimizer_colab_t4_scale/`), two measurement suites (`macbook_mps_gpt2/`, `colab_t4_lite/`), two search pilots (`optimizer_macbook/`, `optimizer_colab_t4/`), and one T4 llama.cpp backend study (`colab_t4_gguf/`). These are separate published directories, not seven independent replications of one experiment. No Llama-3-8B, Mistral, or Qwen 7B+ timings. No energy scores. No MMLU / GSM8K / HumanEval.

---

## 4. Related Work

**Serving systems.** vLLM uses PagedAttention to reduce KV-cache fragmentation and improve throughput [1]. TensorRT-LLM and FasterTransformer similarly assume a CUDA compilation stack. InferLite does not reimplement those kernels. On this Mac, vLLM is unsupported (`PagedAttention is only measured with vLLM, which is not available here`). Continuous batching and speculative decoding in the MPS suite are **in-process research loops**, not vLLM.

**Quantization.** GPTQ [3], AWQ [2], LLM.int8() [4], and QLoRA/NF4 [5] are the methods InferLite *attempts* when the loader exists. SmoothQuant [10] and SqueezeLLM [11] are named in the Mac quantization suite and returned `unsupported` (not bundled). llama.cpp GGUF is a separate backend [12]; its memory snapshot is not CUDA VRAM (Section 13.3).

**Speculative decoding.** Leviathan et al. [13] established draft-target verification. The MPS suite times DistilGPT-2 as draft against GPT-2 with $\gamma \in \{2,4\}$. Acceptance is $0.68$–$0.75$; wall-clock throughput is **lower** than the greedy baseline (77–98 vs 164 tok/s). That is a measured slowdown, not a $4.34\times$ playground claim.

**Auto-tuning and random search.** Bergstra and Bengio [6] showed random search is a strong baseline when only a few hyperparameters matter. Sequential model-based optimization (EGO / Bayesian optimization) treats expensive black-box evaluations as the scarce resource [7, 8]. Multi-objective comparison often uses hypervolume [14]. InferLite is a small ridge surrogate plus a hand-designed acquisition, not GP-EI or NSGA-II [15]. We therefore treat **random search at the same budget** as the primary competitor.

**This work vs those systems.** InferLite is complementary: it decides *what can run here* and *which of those configs to time*, rather than assuming the serving stack is already installed.

---

## 5. Problem Formulation

Let a configuration be
$$
x = (q, c, n, b) \in \mathcal{X},
$$
where $q$ is a method/precision (e.g. `fp16`, `int4_bnb`), $c$ is prompt context length in tokens, $n$ is `max_new_tokens`, and $b$ is batch size. Hardware and installed libraries induce a **feasible set** $\mathcal{X}_{\mathrm{hw}} \subseteq \mathcal{X}$ via a capability probe. InferLite enumerates $\mathcal{X}_{\mathrm{hw}}$ and **does not assign scores** to $x \notin \mathcal{X}_{\mathrm{hw}}$.

A wall-clock evaluator $f$ maps $x$ to
$$
f(x) = \bigl(L_{95}(x),\; M(x),\; T(x)\bigr)
$$
when the load and generate succeed (`status=measured`), where $L_{95}$ is end-to-end P95 latency (ms), $M$ is peak memory (MB; CUDA allocated bytes on T4, RSS on MPS), and $T$ is mean tokens/s. If load fails for a known capability reason, $f(x)$ is undefined and the record is `unsupported`. If an unexpected exception occurs, the record is `error`. In both failure cases metrics are **null**.

We seek the Pareto front of measured points under
$$
\min \; L_{95}(x),\; M(x) \qquad \max \; T(x).
$$
A point $x$ dominates $y$ if it is no worse on all three (after orienting $T$ as $-T$) and strictly better on at least one (`pareto.dominates`).

**Optimizer comparison metric.** Search quality is the 2-D hypervolume in the $(M, T)$ plane, not the 3-D volume that would include $L_{95}$. Implementation: `hypervolume_throughput_memory` in `backend/research/experiments/pareto.py`. Let $P$ be the non-dominated subset of measured $(M_i, T_i)$, sorted by increasing memory. With reference $T_{\mathrm{ref}}=0$ and
$$
M_{\max} = 1.1 \cdot \max_i M_i + 1,
$$
$$
\mathrm{HV}(P) = \sum_{i=1}^{|P|} \max(M_{i+1}-M_i, 0) \cdot \max(T_i - T_{\mathrm{ref}}, 0),
$$
where $M_{|P|+1} := M_{\max}$. Reported tables use the **ratio to the exhaustive grid**
$$
\mathrm{HV}_{\mathrm{rel}}(S) = \frac{\mathrm{HV}(S)}{\mathrm{HV}(\mathrm{grid})}.
$$
A strategy with budget $B \ll |\mathcal{X}_{\mathrm{hw}}|$ is useful if $\mathrm{HV}_{\mathrm{rel}}$ is close to $1$ with low seed variance.

**Replay.** Grid search measures every $x \in \mathcal{X}_{\mathrm{hw}}$ once. Random, InferLite, and the heuristic then evaluate candidates by **cache lookup** of those wall-clock records. InferLite’s sequential picks still depend on which points it has already chosen; the metrics of a pick are never invented.

---

## 6. Methodology

### 6.1 Measurement engine

`research.engine.run_benchmark` loads via `try_load`, discards `warmup_runs` generates, then records `measure_runs` greedy decodes (`temperature=0`, fixed seed). Metrics: TTFT, inter-token latency, end-to-end latency (mean, std, P50/P95/P99), tokens/s, load time, peak RSS and/or CUDA allocated/reserved bytes. Repeated-run 95% CIs on a single config use a normal approximation $\bar x \pm 1.96\,\mathrm{SE}$ when $n_{\mathrm{runs}}\ge 2$ (`percentile_stats`). Missing samples yield `None`, not zeros.

Energy: `probe_energy` returns NVML instant watts if `pynvml` works; otherwise `supported=false`. Energy is **not** part of hypervolume. The Mac scale study records `energy.supported=false` (`no NVML/power sensor on this machine; energy is not scored`). The T4 scale study records `energy.supported=true` with an NVML instant-watt snapshot in `search_study.json`, but each job’s `metrics.energy_j` is **null**, so joules are not scored.

Quality: optional short-passage perplexity exists in the schema. MMLU, GSM8K, and HumanEval are not run and not fabricated.

### 6.2 Status labels

| Status | Meaning | Metrics |
|--------|---------|---------|
| `measured` | Load and generate completed | Filled from samples |
| `unsupported` | Capability or loader refused the method | Always `null` |
| `error` | Unexpected exception | Always `null` |

Unsupported rows appear in catalog figures (`unsupported_experiments.png` in measurement suites). They are **excluded** from Pareto and hypervolume.

### 6.3 Simulation disclaimer

`python -m research simulate` is a Poisson **queueing** model for capacity planning. It is labeled as simulation in the CLI. It is not an LLM benchmark and is not stored under `docs/results/`.

---

## 7. System Architecture

```
YAML config → research.runner
                 ├─ capabilities / environment probe
                 ├─ engine.run_benchmark  (or unsupported)
                 ├─ optimizer (grid / random / heuristic / InferLite)
                 ├─ predictor + ablation
                 └─ JSON / CSV / figures
```

Code path: `configs/*.yaml` → `backend/research/runner.py` → `engine.py`, `optimizer.py`, `predictor.py`, `viz.py`. CLI: `python -m research suite|optimize|predict|capabilities`. Colab: `notebooks/inferlite_colab.ipynb` with `backend/requirements-colab.txt` only (never the full Mac `requirements.txt`, which reinstalls torch and fills the disk).

Method-level model reuse: `make_eval_fn` can keep a loaded `fp16`/`fp32` (or T4 `fp16`/`int4_bnb`) resident. T4 scale uses `keep_one_method: true` so FP16 and INT4 are not coresident in Colab RAM.

---

## 8. Search Space

`search_space.enumerate_space` takes the Cartesian product of methods, context lengths, decode lengths, and batch sizes, then drops methods whose capability key is unsupported (`skip_unsupported: true`).

**MPS scale** (`configs/optimizer_macbook_scale.yaml`):
$$
\{\mathrm{fp32},\mathrm{fp16}\} \times \{32,48,64,96,128\} \times \{8,12,16,24\} \times \{1\} = 40.
$$
Grid: **40 measured, 0 unsupported, 0 error** (2026-08-25 19:41 UTC). `warmup_runs=1`, `measure_runs=2`.

**T4 scale** (`configs/optimizer_colab_t4_scale.yaml`):
$$
\{\mathrm{fp16},\mathrm{int4\_bnb}\} \times \{32,48,64,96,128\} \times \{8,12,16\} \times \{1\} = 30.
$$
Grid: **30 measured, 0 unsupported, 0 error** (2026-08-25 20:08 UTC). `warmup_runs=1`, `measure_runs=1`. First FP16 row is $\approx 33$ tok/s (not the $n=4$ cold start).

**Pilots.** MPS $n=8$: fp32/fp16 × context $\{32,96\}$ × $n\in\{8,16\}$. T4 $n=4$: fp16/INT4 × context $\{32,64\}$ × $n=8$, `warmup_runs=0`.

INT8, AWQ, GPTQ, GGUF, SmoothQuant, SqueezeLLM, and vLLM are **outside** these optimizer spaces because they were unsupported on the corresponding machine (or reserved for a separate GGUF study). The optimizer therefore never “beats” AWQ by leaving it unmeasured.

---

## 9. Multi-Objective Optimization

### 9.1 Grid search

Evaluates every $x \in \mathcal{X}_{\mathrm{hw}}$. Defines $\mathrm{HV}(\mathrm{grid})$ and the reference Pareto set. Cost is $|\mathcal{X}_{\mathrm{hw}}|$ wall-clock jobs.

### 9.2 Random search

Uniform sample of $B$ distinct candidates (`random.Random(seed).sample`). No surrogate. Strong when the front is easy to hit by chance [6] and when $B$ is a large fraction of $N$.

### 9.3 Hardware heuristic

`hardware_heuristic` picks **one** configuration from environment facts, not from measured scores:

- CUDA and GPU memory $< 10\,\mathrm{GiB}$ and `int4_bnb` in the space → prefer INT4;
- else CUDA or MPS and `fp16` present → prefer FP16;
- else FP32 if present.

Among methods with that precision it takes the **median** $(c,n,b)$ after sorting. On MPS this is a mid-context FP16 point (HV$_{\mathrm{rel}}=0.177$ on $n=40$). On T4 scale, GPU memory is $14912.69$ MB (`Tesla T4`), which is not $<10\,000$ MB, so the INT4 branch does not fire and the heuristic prefers FP16 (HV$_{\mathrm{rel}}=0.188$). The $n=4$ T4 pilot README reports the heuristic as “FP16, longer context” at $0.22\times$. The heuristic is a **constant** across budgets because it always uses one evaluation.

### 9.4 InferLite

Budget $B$:

1. Take $\min(3,B)$ **diverse** seeds (`_diverse_seed`: maximize a discrete distance in context, new tokens, method, batch).
2. Force-include the hardware heuristic if it is not already in the seed set (replace the first seed).
3. While fewer than $B$ points have been evaluated, rank remaining candidates with `inferlite_order`.

If fewer than **three measured** points exist, `inferlite_order` returns a diverse shuffle — **no ridge fit**. Thus **budget 2 is not surrogate search**; it is diverse/heuristic seeding. That design fact is stored in `budget_sweep.json` notes.

With $\ge 3$ measured rows, InferLite fits independent ridge models for $(L_{95}, T, M)$ (`PerformancePredictor`, $\ell_2=10^{-2}$, intercept unregularized). Each unevaluated $x$ is scored by:

- whether the **predicted** point lies on the Pareto front of $\{\text{measured}\} \cup \{\hat f(x)\}$ (bonus $10$ if yes);
- plus $L_1$ distance in $(M,T)$ to the nearest measured point (diversity).

Predicted metrics are tagged `search_prediction` and **never written as measured**. The next evaluation is always `run_benchmark` (or a cache hit of a prior measurement).

Acquisition is greedy and myopic. There is no expected hypervolume improvement [14], no GP uncertainty, and no evolutionary search [15].

---

## 10. Baselines

All four strategies share $\mathcal{X}_{\mathrm{hw}}$ and the same cached $f(x)$. Fairness properties:

| Strategy | Evaluations | Uses $f$ | Uses surrogate |
|----------|-------------|----------|----------------|
| Grid | $N$ | yes | no |
| Random | $B$ | yes | no |
| Heuristic | $1$ | yes (after the pick) | no |
| InferLite | $B$ | yes | yes, only after 3 measured points |

The primary scientific comparison is **InferLite vs random at equal $B$**. Grid is the oracle. The heuristic is a cheap control, not a tuned SOTA policy.

---

## 11. Performance Predictor

Independent ridge regressions for $L_{95}$, $T$, and $M$ (`predictor.py`). Features:

- **Quantization:** one-hot method.
- **Hardware:** CUDA flag, MPS flag, `gpu_mem_mb` (constant on a single machine).
- **Workload:** $\log(1+c)$, $\log(1+n)$, $\log(1+b)$.

Validation is **leave-one-out** on measured rows only. Ablations drop one feature group at a time. $R^2 < 0$ means worse than predicting the training mean; we report it.

The predictor is used in two roles: (i) InferLite acquisition, (ii) a post-hoc ablation on the full grid to see which features carry which targets. Role (ii) is not a claim that the surrogate generalizes to a new GPU.

---

## 12. Experimental Setup

These are **seven measured study artifacts** across two hardware environments, not seven independent replications of one experiment: two measurement suites, two scale-search studies, two search pilots, and one T4 llama.cpp backend study.

| Study | Hardware | Model | $N$ / notes | Artifacts |
|-------|----------|-------|-------------|-----------|
| Measurement suite | Apple MPS | GPT-2 / DistilGPT-2 | 29 records: 19 measured, 10 unsupported, 0 error | `docs/results/macbook_mps_gpt2/` |
| T4 lite | Colab Tesla T4 (14.9 GB) | TinyLlama 1.1B HF | 8 records: 4 measured, 4 unsupported, 0 error | `docs/results/colab_t4_lite/` |
| T4 llama.cpp | same T4 session | TinyLlama Q4_K_M GGUF | 1 measured | `docs/results/colab_t4_gguf/` |
| Search scale | Apple MPS | GPT-2 | **40** configs; 5 seeds; $B\in\{2,4,8,16\}$ | `docs/results/optimizer_macbook_scale/` |
| Search scale | Tesla T4 | TinyLlama HF | **30** configs; same seeds/budgets; `warmup_runs=1` | `docs/results/optimizer_colab_t4_scale/` |
| Search pilot | Apple MPS | GPT-2 | 8 configs; seed 42; $B=4$ | `docs/results/optimizer_macbook/` |
| Search pilot | Tesla T4 | TinyLlama HF | 4 configs; seed 42; $B=2$; `warmup_runs=0` | `docs/results/optimizer_colab_t4/` |

**Not timed / not scored as objectives:** Llama-3 / Mistral / Qwen 7B+, energy joules (`energy_j` null even when NVML probes watts on T4; Mac has no NVML), MMLU / GSM8K / HumanEval, Mac→T4 predictor transfer, vLLM, TensorRT-LLM, AWQ/GPTQ as real quantized checkpoints on these runs.

**Seeds (scale):** $42, 123, 456, 789, 1000$. **t-critical value** for $n=5$ ($df=4$) is $2.776$ (`metrics._T_CRIT_975`). Interval:
$$
\bar x \pm t_{0.975,\,n-1}\, s / \sqrt{n}, \quad s = \text{sample std (ddof=1)}.
$$
HV ratios themselves are bounded by the exhaustive-grid reference ($\mathrm{HV}_{\mathrm{rel}}\le 1$ for a subset of the same measured points). **Confidence intervals may extend above 1** (Mac $B=16$ InferLite $[0.930, 1.011]$; T4 $B=16$ InferLite $[0.988, 1.001]$) **because they quantify uncertainty around the sample mean; the normalized hypervolume itself is bounded by the exhaustive-grid reference.** Intervals may also go below 0 when the seed-to-seed standard deviation is large and $n=5$; that is a poorly calibrated $t$-interval, not negative hypervolume.

**Decode:** greedy, seed 42 inside `run_benchmark` for a given measurement. Optimizer seeds shuffle *which* configs are chosen, not the generate RNG independently per strategy beyond that.

---

## 13. Results

Different models and backends must not be stacked into one ranking.

### 13.1 Measurement suite: MacBook MPS / GPT-2

Config `configs/macbook_cpu.yaml` (device in the CSV is `mps`, not CPU). Prompt: *“The future of efficient language model inference is”*; quantization jobs use `max_new_tokens=24`, `warmup_runs=1`, `measure_runs=3`. Records: **29** (19 measured, 10 unsupported, 0 error).

**Quantization (measured).** FP32: $145.8$ tok/s, P95 $174.8$ ms, RSS $761$ MB. FP16: $211.2$ tok/s, P95 $118.3$ ms, RSS $1064$ MB. FP16 is the dense Hugging Face throughput win on this MPS run. Figure 1 shows only those two measured quantization bars; unsupported methods are omitted from the plot, not scored as zero.

![Figure 1. Measured GPT-2 quantization on Apple MPS (`macbook_cpu`): mean tokens/s (left) and end-to-end P95 (right). Source: `docs/results/macbook_mps_gpt2/figures/quantization_comparison.png`.](results/macbook_mps_gpt2/figures/quantization_comparison.png)

**Quantization-family and KV methods that did not load (9 + PagedAttention = 10 `unsupported` rows):** bf16 (CUDA required); dynamic INT8 (`linear_prepack` / NoQEngine); bitsandbytes INT8 and INT4 (CUDA); AWQ; GPTQ; GGUF (`llama_cpp` missing); SmoothQuant (not wired); SqueezeLLM (not bundled); `paged_attention` (no vLLM). These are **not** ranked last. They have no tokens/s. Figure 2 is the catalog of those ten `unsupported` rows.

![Figure 2. Unsupported experiments in the MPS measurement suite (not scored). Ten rows, all `status=unsupported`, zero `error`. Source: `docs/results/macbook_mps_gpt2/figures/unsupported_experiments.png`.](results/macbook_mps_gpt2/figures/unsupported_experiments.png)

**KV cache (measured, FP16).** Dynamic cache at context $32/64/128$: $136$ / $128$ / $115$ tok/s. `no_cache` is slower at short context ($97$ and $72$ tok/s at $c=32,64$; $87$ tok/s at $c=128$). `sliding_window` is $\approx 231$–$235$ tok/s at all three lengths because it is **prompt truncation**, not a fused sliding-window kernel. `prefix` rows are measured but tokens/s is empty in the CSV (not used as throughput scores). Figure 3 plots **peak RSS and TTFT**, not tokens/s: `dynamic` is the lowest RSS; `sliding_window` RSS is flat because the prompt is truncated.

![Figure 3. KV-cache scaling on MPS: peak RSS vs context (left) and mean TTFT vs context (right). Tokens/s for these rows are in `experiments.csv`, not this figure. Source: `docs/results/macbook_mps_gpt2/figures/kv_cache_scaling.png`.](results/macbook_mps_gpt2/figures/kv_cache_scaling.png)

**Speculative decoding.** Baseline greedy: $164.0$ tok/s. $\gamma=2$: $77.4$ tok/s, acceptance $0.75$, speedup vs baseline $0.47\times$. $\gamma=4$: $98.2$ tok/s, acceptance $0.68$, speedup $0.60\times$. Draft acceptance is real; wall-clock speedup is **not**.

![Figure 4. Speculative decoding on MPS: wall-clock tokens/s (left) and draft acceptance (right). Baseline is faster than $\gamma=2$ and $\gamma=4$. Source: `docs/results/macbook_mps_gpt2/figures/speculative_decoding.png`.](results/macbook_mps_gpt2/figures/speculative_decoding.png)

**Batching.** `static_batch` $53.1$ tok/s vs `continuous_batch` $41.4$ tok/s (in-process loop, 4 requests, max batch 2, 8 new tokens). Not vLLM continuous batching.

Figure 5 is the suite’s measured throughput–memory Pareto (quantization + KV rows that have tokens/s). `fp32`, `fp16`, and `sliding_window` lie on the drawn front; `dynamic` and `no_cache` do not. `sliding_window` is prompt truncation, so that high-throughput point is **not** a fused-kernel result.

![Figure 5. MPS measurement-suite throughput–memory Pareto (measured points with tokens/s only). Source: `docs/results/macbook_mps_gpt2/figures/pareto_throughput_memory.png`.](results/macbook_mps_gpt2/figures/pareto_throughput_memory.png)

### 13.2 Measurement suite: Colab T4 / TinyLlama Hugging Face

Lite config. **Measured Pareto (notebook stdout):**

| Method | tok/s | P95 e2e (ms) | GPU mem (MB) |
|--------|------:|-------------:|-------------:|
| fp16 | 34.975 | 467.92 | 2107.5 |
| int4_bnb | 15.306 | 1056.69 | 802.7 |

FP16 wins throughput; INT4 wins memory. Figure 6 also shows `int8_bnb` and a bar labeled `gptq`. INT8 is $\approx 3.0$ tok/s / $\approx 5.4$ s P95 (figure-derived, not an engine CSV on this machine). The `gptq` bar is **dense TinyLlama** (no `gptq_model_id`); it is not a GPTQ checkpoint measurement ($\approx 19.5$ tok/s / $\approx 860$ ms, also figure-derived). Do not cite those clocks as GPTQ.

![Figure 6. T4 lite quantization bars. The `gptq` label is dense TinyLlama, not GPTQ. Source: `docs/results/colab_t4_lite/figures/quantization_comparison.png`.](results/colab_t4_lite/figures/quantization_comparison.png)

Unsupported in that lite suite: AWQ (`autoawq` missing), GGUF (`llama_cpp` missing *at that time*), SmoothQuant, SqueezeLLM. Zero errors (Figure 7). GGUF was measured later in the same Colab session (`docs/results/colab_t4_gguf/`).

![Figure 7. Unsupported methods in the T4 lite suite (not scored). Source: `docs/results/colab_t4_lite/figures/unsupported_experiments.png`.](results/colab_t4_lite/figures/unsupported_experiments.png)

Figure 8 is the lite throughput–memory plot. The drawn front is **fp16** and **int4_bnb**. `int8_bnb` and the dense-`gptq` bar sit off the front. This plot is Hugging Face Transformers only; do not overlay llama.cpp (Section 13.3).

![Figure 8. T4 lite throughput–memory Pareto (measured Hugging Face points). Source: `docs/results/colab_t4_lite/figures/pareto_throughput_memory.png`.](results/colab_t4_lite/figures/pareto_throughput_memory.png)

### 13.3 llama.cpp GGUF on the same T4

Separate study after installing `llama-cpp-python` 0.3.35 from the CUDA 12.4 wheel (`--only-binary=:all:`). Checkpoint: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, `n_gpu_layers=-1`.

| Backend | tok/s | P95 (ms) | Memory |
|---------|------:|---------:|--------|
| llama.cpp Q4_K_M | 172.500 | 105.769 | 9.125 MB **engine snapshot, not VRAM** |

Do not compare 9.125 MB to Hugging Face 2108 MB CUDA as if they were the same allocator. Do not mix 172.5 tok/s into the bitsandbytes Pareto without labeling the backend.

### 13.4 Optimizer scale: MPS, $N=40$, five seeds

This is the strongest **positive** InferLite-vs-random comparison in the repository. It is **not** a license to claim InferLite always beats random: Section 13.5 reports the opposite ranking on T4 at the same $B=4$. Together the two tables are the result: **the effectiveness of surrogate-guided inference optimization is hardware- and budget-dependent.**

Grid HV $= 62165.22$ (absolute, memory×throughput units). Figure 9 is the main hypervolume-vs-budget plot.

![Figure 9. Mean HV relative to exhaustive grid vs evaluation budget on GPT-2 / Apple MPS ($N=40$). Error bars: 95% Student-$t$ intervals over seeds $\{42,123,456,789,1000\}$. Grid is the dashed line at 1.0. Heuristic is one evaluation ($\mathrm{HV}_{\mathrm{rel}}=0.177$).](results/optimizer_macbook_scale/figures/hv_vs_budget.png)

| $B$ | InferLite mean (std) [95% CI] | Random mean (std) [95% CI] | InferLite mean $>$ random? |
|----:|------------------------------:|---------------------------:|:---------------------------|
| 2 | 0.481 (0.244) [0.178, 0.784] | 0.310 (0.363) [−0.140, 0.760] | yes |
| 4 | 0.865 (0.031) [0.826, 0.903] | 0.607 (0.395) [0.117, 1.097] | yes |
| 8 | 0.922 (0.021) [0.895, 0.948] | 0.800 (0.379) [0.330, 1.270] | yes |
| 16 | 0.971 (0.032) [0.930, 1.011] | **0.973 (0.015) [0.954, 0.991]** | **no** |

Source: `docs/results/optimizer_macbook_scale/hv_vs_budget.csv`.

At $B=4$ (4 of 40 evaluations, **10% of the grid**) InferLite achieved a **higher mean** $\mathrm{HV}_{\mathrm{rel}}$ than random ($0.86\times$ vs $0.61\times$) and was **much more stable** (std $0.03$ vs $0.39$). This is a comparison of means over five seeds, **not** a claim of statistical significance. Per-seed $\mathrm{HV}_{\mathrm{rel}}$ at $B=4$: InferLite $\{0.829, 0.906, 0.888, 0.850, 0.850\}$; random $\{0.107, 0.818, 0.257, 0.904, 0.950\}$ (`budget_sweep.json`). Random’s interval includes values $>1$ and, at $B=2$, $<0$: the $t$-interval is poorly calibrated when $s$ is large and $n=5$.

At $B=16$ ($\approx 40\%$ of the space) both methods sit near the grid; random is slightly ahead on the mean. InferLite does **not** replace exhaustive search once the budget is large.

![Figure 10. MPS scale at the highlight budget $B=4$: wall-clock evaluations (left) and mean $\mathrm{HV}_{\mathrm{rel}}$ (right). InferLite and random both use 4 evaluations; grid uses 40; heuristic uses 1. Source: `docs/results/optimizer_macbook_scale/figures/search_vs_baselines.png`.](results/optimizer_macbook_scale/figures/search_vs_baselines.png)

Figure 11 is the **exhaustive grid** in the $(M,T)$ plane (all 40 measured points). FP32 clusters near $\approx 440$–$470$ MB RSS; FP16 near $\approx 740$–$760$ MB. On this MPS run FP16 uses **more** RSS than FP32 and reaches higher tokens/s. The red line is the measured throughput–memory Pareto of the grid, not of InferLite’s budget-4 subset.

![Figure 11. MPS $N=40$ grid: throughput vs peak RSS, measured only. Source: `docs/results/optimizer_macbook_scale/figures/pareto_throughput_memory.png`.](results/optimizer_macbook_scale/figures/pareto_throughput_memory.png)

### 13.5 Optimizer scale: T4, $N=30$, five seeds

Same protocol, opposite ranking at small budgets. At $B=4$, random achieved a **higher mean** $\mathrm{HV}_{\mathrm{rel}}$ than InferLite ($0.96\times$ vs $0.49\times$). That contrast with Section 13.4 is the scientific result, not a defect to hide.

Grid HV $= 35888.12$. `warmup_runs=1`. First published FP16 scale row is $33.4$ tok/s at $c=32$, $n=8$ (not $6.1$ tok/s).

![Figure 12. Same protocol on TinyLlama / Tesla T4 ($N=30$). Random has a higher mean $\mathrm{HV}_{\mathrm{rel}}$ than InferLite at $B=2$ and $B=4$; the curves meet near $0.97$ at $B=8$.](results/optimizer_colab_t4_scale/figures/hv_vs_budget.png)

| $B$ | InferLite mean (std) [95% CI] | Random mean (std) [95% CI] | InferLite mean $>$ random? |
|----:|------------------------------:|---------------------------:|:---------------------------|
| 2 | 0.329 (0.305) [−0.049, 0.708] | **0.611 (0.449) [0.054, 1.168]** | no |
| 4 | 0.490 (0.403) [−0.010, 0.991] | **0.957 (0.022) [0.930, 0.985]** | no |
| 8 | **0.971 (0.032) [0.931, 1.010]** | 0.967 (0.021) [0.941, 0.994] | yes |
| 16 | **0.995 (0.005) [0.988, 1.001]** | 0.990 (0.011) [0.976, 1.004] | yes |

Heuristic: $0.188$ at every budget.

**Why random wins at $B=4$ on T4.** InferLite per-seed $\mathrm{HV}_{\mathrm{rel}}$: $\{0.198, 0.984, 0.198, 0.875, 0.198\}$. Three of five seeds remain near the heuristic-scale front ($\approx 0.20$). Random is tightly clustered: $\{0.965, 0.946, 0.937, 0.992, 0.945\}$. On this TinyLlama space a uniform sample of four points almost always covers both fp16 (high $T$, high $M$) and int4 (low $M$). InferLite’s diverse/heuristic seed plus one surrogate step is **seed-fragile** here. That is a real limitation of the acquisition, not a plotting artifact.

Do not copy Figure 9’s ranking onto Figure 12.

Figure 13 is the highlight-budget bar chart ($B=4$): random $\approx 0.96\times$, InferLite $\approx 0.49\times$, heuristic $\approx 0.19\times$, grid $1.0$ at 30 evaluations. This is the same ranking as the table, not a different experiment.

![Figure 13. T4 scale at $B=4$: wall-clock evaluations (left) and mean $\mathrm{HV}_{\mathrm{rel}}$ (right). Random’s bar is near the grid; InferLite’s is not. Source: `docs/results/optimizer_colab_t4_scale/figures/search_vs_baselines.png`.](results/optimizer_colab_t4_scale/figures/search_vs_baselines.png)

Figure 14 is the $N=30$ grid in the $(M,T)$ plane. INT4 (`int4_bnb`) clusters near $\approx 800$–$850$ MB CUDA allocated; FP16 near $\approx 2110$ MB. That two-cluster geometry is why a uniform sample of four points often hits both sides of the front.

![Figure 14. T4 $N=30$ grid: throughput vs peak CUDA allocated memory, measured only. Source: `docs/results/optimizer_colab_t4_scale/figures/pareto_throughput_memory.png`.](results/optimizer_colab_t4_scale/figures/pareto_throughput_memory.png)

### 13.6 Pilots (kept as negative controls)

**MPS $n=8$, seed 42, $B=4$** (`docs/results/optimizer_macbook/comparison.json`):

| Strategy | Evals | $\mathrm{HV}_{\mathrm{rel}}$ |
|----------|------:|-----------------------------:|
| Grid | 8 | 1.000 |
| Random | 4 | **0.324** |
| InferLite | 4 | 0.298 |
| Heuristic | 1 | 0.225 |

On this seed random slightly beat InferLite. Neither recovered the grid front. Cite Section 13.4, not this ranking, for the optimizer claim.

![Figure 15. MPS $n=8$ pilot, seed 42. Source: `docs/results/optimizer_macbook/figures/search_vs_baselines.png`.](results/optimizer_macbook/figures/search_vs_baselines.png)

**T4 $n=4$, seed 42, $B=2$, `warmup_runs=0`:**

| Candidate | tok/s | P95 (ms) | GPU MB | Note |
|-----------|------:|---------:|-------:|------|
| fp16 $c=32$ | 6.125 | 1306.2 | 2110 | cold start |
| fp16 $c=64$ | 31.824 | 251.4 | 2114 | later FP16 job |
| int4 $c=32$ | 18.717 | 427.4 | 804 | |
| int4 $c=64$ | 15.370 | 520.5 | 827 | |

| Strategy | Evals | $\mathrm{HV}_{\mathrm{rel}}$ |
|----------|------:|-----------------------------:|
| Grid | 4 | 1.000 |
| InferLite | 2 | **0.849** |
| Heuristic | 1 | 0.216 |
| Random | 2 | 0.041 |

InferLite evaluated `fp16|c64` and `int4|c64`. Random drew the cold-start `fp16|c32` point. $n=4$ is too small, and the first row is not steady-state FP16. Do not mix with Section 13.5.

![Figure 16. T4 $n=4$ pilot (cold-start first FP16 job). Source: `docs/results/optimizer_colab_t4/figures/search_vs_baselines.png`.](results/optimizer_colab_t4/figures/search_vs_baselines.png)

---

## 14. Ablation Studies

Leave-one-out ridge on the **full measured grid** (not on InferLite’s $B$ points). Source JSON: `docs/results/optimizer_*_scale/ablation.json`.

### 14.1 MPS, $n=40$

![Figure 17. Leave-one-out tokens/s $R^2$ when dropping feature groups (MPS $n=40$). Negative bars mean worse than predicting the mean.](results/optimizer_macbook_scale/figures/predictor_ablation.png)

| Variant | P95 $R^2$ | Tokens/s $R^2$ | Memory $R^2$ | P95 MAE | Tok/s MAE | Mem MAE |
|---------|----------:|---------------:|-------------:|--------:|----------:|--------:|
| Full | 0.711 | 0.607 | 0.9993 | 25.1 ms | 21.0 | 2.80 MB |
| No hardware | 0.711 | 0.607 | 0.9993 | 25.1 | 21.0 | 2.80 |
| No quantization | 0.592 | **−0.172** | **−0.171** | 29.4 | 36.4 | 158 |
| No workload | **0.017** | 0.644 | 0.9975 | 45.6 | 20.0 | 5.69 |

Hardware features are constant on one Mac, so dropping them changes nothing. Quantization (fp16 vs fp32) carries memory and throughput. Workload (context and new tokens) carries P95.

### 14.2 T4, $n=30$

![Figure 18. Same ablation on TinyLlama / T4 ($n=30$).](results/optimizer_colab_t4_scale/figures/predictor_ablation.png)

| Variant | P95 $R^2$ | Tokens/s $R^2$ | Memory $R^2$ | P95 MAE | Tok/s MAE | Mem MAE |
|---------|----------:|---------------:|-------------:|--------:|----------:|--------:|
| Full | 0.843 | 0.678 | 0.9999 | 59.4 ms | 2.75 | 4.75 MB |
| No hardware | 0.843 | 0.678 | 0.9999 | 59.4 | 2.75 | 4.75 |
| No quantization | 0.476 | **−0.223** | **−0.240** | 115 | 6.38 | 721 |
| No workload | 0.208 | 0.712 | 0.9998 | 140 | 2.60 | 7.74 |

Same qualitative story. Memory MAE jumps from $5$ MB to $721$ MB without the method one-hot: the model can no longer separate $\approx 2110$ MB FP16 from $\approx 830$ MB INT4.

### 14.3 Pilots

**MPS $n=8$:** full P95 $R^2=0.86$, tokens/s $0.51$, memory $0.9995$. No quantization: tokens/s $-1.26$, memory $-1.52$. No workload: P95 $-0.72$.

**T4 $n=4$:** full P95 $R^2=-6.66$, tokens/s $-8.44$, memory $0.999$. The predictor **fails to generalize with four T4 observations**, including a cold-start outlier. This is a dataset limit. It is not corrected in the $n=4$ table. The $n=30$ fit is the T4 predictor to cite.

---

## 15. Statistical Analysis

**Design.** For each hardware, one exhaustive measured grid; $5$ independent seeds; $4$ budgets. Strategies share $f(x)$. Variance is over seeds, not over repeated generates of the same config (those repeats are already averaged inside `measure_runs`).

**Interval.** Student-$t$ with $df=4$, $t_{0.975}=2.776$. We report mean, unbiased std, and CI of $\mathrm{HV}_{\mathrm{rel}}$. Five seeds is a **small** sample. We do **not** run paired Wilcoxon or $t$-tests in the repository and we do **not** call these rankings statistically significant. On MPS $B=4$, InferLite’s CI $[0.83, 0.90]$ does not overlap random’s mean $0.61$, but random’s CI is so wide $[0.12, 1.10]$ that it *does* overlap InferLite’s mean. The honest summary is: InferLite achieved a **higher mean** hypervolume and was more **stable**; a test of means at $n=5$ is underpowered for random.

**Intervals above 1.** Confidence intervals may extend above 1 because they quantify uncertainty around the sample mean; the normalized hypervolume itself is bounded by the exhaustive-grid reference. Mac $B=16$ InferLite $[0.930, 1.011]$ and T4 $B=16$ InferLite $[0.988, 1.001]$ are examples, not evidence that a subset beat the grid.

**Budget 2.** InferLite does not fit a ridge model. Comparing “InferLite” to random at $B=2$ is comparing diverse/heuristic seeding to uniform sampling.

**Multiple hardware.** The MPS and T4 rankings disagree at $B=4$. Pooling them into one “InferLite wins” sentence would be false. The scientific claim is that **surrogate-guided inference optimization is hardware- and budget-dependent**. InferLite does not always beat random. The research question is **when** it does: *on this space, at this $B$, on this device*.

**Replay vs re-measure.** Replay removes load-time noise from strategy comparisons. It also means InferLite never observes a fresh noisy $f(x)$ that differs from the grid. That is closer to “noiseless BO on a lookup table” than to online auto-tuning under nonstationarity.

---

## 16. Limitations

**Models and devices.** GPT-2 (124M) and TinyLlama 1.1B on one Mac MPS and one Colab T4. No 7B+ Llama/Mistral/Qwen, no A100/H100, no RTX 4060. Serving stacks that matter in production (vLLM, TensorRT-LLM) are unsupported here.

**Search space size.** $N=40$ and $N=30$ are larger than the $n=8$/$n=4$ pilots but still tiny versus industrial auto-tuning. Random’s success on T4 at $B=4$ is easier when half the space is “the other precision.”

**Acquisition.** Hand-designed score; no uncertainty; ridge with $\ell_2=10^{-2}$; three-point minimum before the surrogate exists. Three of five T4 seeds at $B=4$ fail to leave a weak front.

**Objectives.** Hypervolume ignores $L_{95}$ even though InferLite’s predictor fits it. Memory on MPS is RSS, not accelerator-allocated bytes. llama.cpp’s $9.125$ MB is an engine snapshot.

**Workloads.** Synthetic token-length prompts (`prompt_for_tokens`), greedy decode, batch $1$ in search. Sliding-window KV is truncation. Speculative and continuous-batching suites are research loops.

**Energy and quality.** Energy is not in hypervolume. Mac scale: NVML unsupported. T4 scale: NVML instant-watt probe only; `energy_j` is null. MMLU / GSM8K / HumanEval are not run.

**Statistics.** $n_{\mathrm{seeds}}=5$ is small; CIs miscalibrated under large $s$; no hypothesis tests and no correction for looking at four budgets × two devices. Do not read mean rankings as statistically significant.

**Generalization.** Hardware features do nothing on a single machine. Cross-device transfer (train on MPS, test on T4) is **not** measured.

**Threats to validity.** (i) *Internal:* replay uses the same $f(x)$ for all strategies — good for fairness, optimistic for online noise. (ii) *Construct:* $\mathrm{HV}_{\mathrm{rel}}$ in $(M,T)$ may prefer high-throughput high-memory points an operator would reject. (iii) *External:* small models and two devices. (iv) *Conclusion:* pilots can reverse the scale ranking (MPS $n=8$ seed 42).

---

## 17. Reproducibility

Artifacts: `docs/results/*/experiments.csv`, `budget_sweep.json`, `hv_vs_budget.csv`, `ablation.json`, `figures/`. Configs: `configs/optimizer_macbook_scale.yaml`, `configs/optimizer_colab_t4_scale.yaml`, `configs/macbook_cpu.yaml`, `configs/colab_t4_lite.yaml`, `configs/colab_t4_gguf.yaml`.

```bash
cd backend
source .venv/bin/activate    # this Mac has no system `python`
python -m research capabilities
python -m research suite --config ../configs/macbook_cpu.yaml
python -m research optimize --config ../configs/optimizer_macbook_scale.yaml
python -m research optimize --config ../configs/optimizer_macbook.yaml
python -m pytest
```

Colab T4: [notebook](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb). Install **only** `backend/requirements-colab.txt`. Clean runtime for search; skip lite and GGUF in the same session if RAM is tight. Confirm `measured 30` for the scale study.

Software: https://github.com/Shivani767/llm-inferlite (Apache 2.0).

---

## 18. Discussion

The effectiveness of surrogate-guided inference optimization is **hardware- and budget-dependent**. InferLite does not always beat random. The research question is *when* InferLite beats random, not a claim that it always does.

The strongest InferLite-vs-random comparison is MPS GPT-2 at $B=4$: mean $\mathrm{HV}_{\mathrm{rel}}=0.86$ using $4/40=10\%$ of exhaustive measurements, versus random $0.61$, with much lower seed variance. That is evidence that a sequential policy *can* approach the front with a limited budget on this space. It is not a significance test.

On T4 TinyLlama at the same $B=4$, random achieved a higher mean hypervolume ($0.96\times$ vs InferLite $0.49\times$). The feasible set is essentially two memory/throughput clusters (fp16 vs INT4). Uniform samples of size 4 almost always hit both clusters; InferLite’s early trajectory does not. Surrogate search is not automatically better than random on a small, clustered space — consistent with Bergstra and Bengio’s warning that adaptive methods must beat random, not merely look more sophisticated [6].

On MPS, random eventually matches InferLite when $B$ is 40% of $N$ ($B=16$: $0.973$ vs $0.971$). That matches the classical view that sequential search can help when evaluations are few, and that random search is competitive as the budget grows [6].

The predictor ablations explain *what can be learned* from a full grid: quantization ≈ memory and tok/s; workload ≈ P95; hardware ≈ nothing on one device. They do **not** imply that InferLite’s online ridge, fit on three points, is well-specified. The $n=4$ T4 negative $R^2$ is the correct warning.

Practically: if the operator can afford $N$ measurements, use the grid. If the operator can afford $\approx N/10$ on a space that looks like MPS GPT-2, InferLite is a reasonable sequential policy **on the evidence we have**. If the space looks like two quantizations × a few contexts on T4, random at $B=4$ was better. Either way, missing kernels stay `unsupported`.

---

## 19. Conclusion

InferLite is a hardware-aware measurement-and-search loop for LLM inference configurations. It times what loads, refuses to score missing kernels, compares grid / random / heuristic / surrogate search on cached wall-clock records, and reports hypervolume vs budget with five seeds.

The headline finding is not that InferLite always beats random. It is that **surrogate-guided inference optimization is hardware- and budget-dependent**. On a 40-point MPS GPT-2 grid, InferLite achieved a higher mean hypervolume than random at 4 evaluations ($0.86\times$ of the exhaustive front, 10% of the measurements) with a tight interval; random was lower on the mean and unstable; at 16 evaluations random was slightly ahead. On a 30-point T4 TinyLlama grid, random was ahead at 4 evaluations ($0.96\times$ vs $0.49\times$) and the methods met near $0.97\times$ at budget 8. An 8-point MPS pilot had random slightly ahead of InferLite. A 4-point T4 predictor had negative P95/tokens/s $R^2$; a 30-point T4 predictor did not.

Those sentences can appear in the same paper because none of the numbers were invented. The open question is not whether to add RAG or a chatbot: it is under which **search-space geometries and budgets** a hardware-aware surrogate beats random search — and that question is only as good as the next measured grid.

---

## References

Verified from public bibliographic sources (arXiv / ACM / JMLR). Items marked *software* are repositories, not peer-reviewed articles.

1. W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient Memory Management for Large Language Model Serving with PagedAttention,” in *Proc. SOSP*, 2023. arXiv:2309.06180. DOI: [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165).

2. J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,” *MLSys*, 2024. arXiv:2306.00978. *[venue year cross-checked against arXiv 2306.00978; confirm camera-ready page numbers if citing in camera-ready form.]*

3. E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,” *ICLR*, 2023. arXiv:2210.17323.

4. T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale,” *NeurIPS*, 2022. arXiv:2208.07339.

5. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” *NeurIPS*, 2023. arXiv:2305.14314. (NF4 / bitsandbytes 4-bit.)

6. J. Bergstra and Y. Bengio, “Random Search for Hyper-Parameter Optimization,” *JMLR*, vol. 13, pp. 281–305, 2012. [https://www.jmlr.org/papers/v13/bergstra12a.html](https://www.jmlr.org/papers/v13/bergstra12a.html).

7. D. R. Jones, M. Schonlau, and W. J. Welch, “Efficient Global Optimization of Expensive Black-Box Functions,” *Journal of Global Optimization*, vol. 13, pp. 455–492, 1998.

8. J. Snoek, H. Larochelle, and R. P. Adams, “Practical Bayesian Optimization of Machine Learning Algorithms,” *NeurIPS*, 2012.

9. P. Zhang, G. Zeng, T. Wang, and W. Lu, “TinyLlama: An Open-Source Small Language Model,” arXiv:2401.02385, 2024.

10. G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models,” *ICML*, 2023. arXiv:2211.10438.

11. S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney, and K. Keutzer, “SqueezeLLM: Dense-and-Sparse Quantization,” arXiv:2306.07629, 2023. *[conference camera-ready metadata not re-fetched for this draft; arXiv id verified as the project’s usual citation.]*

12. G. Gerganov and contributors, *llama.cpp*. Software. [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).

13. Y. Leviathan, M. Kalman, and Y. Matias, “Fast Inference from Transformers via Speculative Decoding,” *ICML*, 2023. arXiv:2211.17192.

14. E. Zitzler and L. Thiele, “Multiobjective Evolutionary Algorithms: A Comparative Case Study and the Strength Pareto Approach,” *IEEE Trans. Evolutionary Computation*, vol. 3, no. 4, pp. 257–271, 1999.

15. K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II,” *IEEE Trans. Evolutionary Computation*, vol. 6, no. 2, pp. 182–197, 2002.

16. A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for Nonorthogonal Problems,” *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

17. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners,” OpenAI technical report, 2019. (GPT-2.) *[not a conference PDF with DOI in this repo; treat as the standard GPT-2 tech report.]*

18. NVIDIA, *TensorRT-LLM*. Software documentation. Not measured in this repository.

19. Hugging Face, *Transformers* / *bitsandbytes* integrations. Software. Used as loaders; not a substitute for [4, 5].

20. S. Bhandari, *InferLite*. Software, 2026. [https://github.com/Shivani767/llm-inferlite](https://github.com/Shivani767/llm-inferlite).
