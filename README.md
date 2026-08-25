# InferLite

**Hardware-aware multi-objective LLM inference optimization**

InferLite searches a configuration space (quantization, context, decode length) on the machine you have, times only what loads, and returns a Pareto front. Missing kernels are **unsupported**, not scored. Result tables are **measured**, never simulated Llama-3-8B / TensorRT numbers.

Paper: [`docs/paper.md`](docs/paper.md) · Raw artifacts: [`docs/results/`](docs/results/)

## Research question

Can we find a strong LLM inference configuration for given hardware and workload **without exhaustive benchmarking** — and without inventing scores for kernels that are not installed?

## Method

1. Probe hardware and libraries (CUDA / MPS / bitsandbytes / AWQ / GPTQ / llama.cpp / vLLM).
2. Enumerate a filtered search space. Drop unsupported methods instead of ranking them last.
3. Wall-clock greedy decode (seed 42). Record TTFT, tokens/s, P50/P95/P99, memory, load time.
4. Compare **grid vs random vs hardware heuristic vs InferLite** (ridge surrogate) on the same measured points. Hypervolume vs exhaustive grid.
5. Fit a predictor (hardware + quantization + workload → P95, tokens/s, memory). Ablate each feature group with leave-one-out R².

**measured** — timed here · **unsupported** — did not run · **error** — attempted and failed

`python -m research simulate` is a Poisson **queueing** model for capacity planning. It is not an LLM benchmark. Do not cite it, the FastAPI advisor, or old Llama-3-8B TensorRT tables as measurements.

## Architecture

```
                 InferLite
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
  Hardware      Workload      Model
       │            │            │
       └────────────┼────────────┘
                    ↓
           Configuration space
                    ↓
         Quantization / runtime / KV
                    ↓
         Multi-objective optimizer
                    ↓
              Pareto frontier
                    ↓
           Best measured config
```

Code path: `configs/*.yaml` → `research.runner` → engine + search + predictor → JSON/CSV + figures.

Centerpiece experiment: **grid search vs random search vs InferLite** (`python -m research optimize`).

## Experimental setup

| | MacBook | Colab |
|--|---------|-------|
| Device | Apple MPS | Tesla T4 |
| Models timed | GPT-2 / DistilGPT-2 | TinyLlama 1.1B (HF + GGUF) |
| Not timed | Llama-3 / Mistral / Qwen 7B+ | Llama-3 / Mistral / Qwen 7B+ |
| Optimizer | 8 configs, budget 4 | 4 configs, budget 2 |

YAML, seed 42, `backend/requirements.txt` (Mac) and `backend/requirements-colab.txt` (Colab only — never full `requirements.txt` on Colab).

## Results

Five **measured** studies. Different models and backends — do not stack the tables. Raw logs live under [`docs/results/`](docs/results/), not in this README.

| Study | Model | Device | Artifacts |
|-------|-------|--------|-----------|
| Measurement suite | GPT-2 | MacBook MPS | [`docs/results/macbook_mps_gpt2/`](docs/results/macbook_mps_gpt2/) |
| T4 lite (Hugging Face) | TinyLlama 1.1B | Colab T4 | [`docs/results/colab_t4_lite/`](docs/results/colab_t4_lite/) |
| T4 llama.cpp | TinyLlama Q4_K_M | Colab T4 | [`docs/results/colab_t4_gguf/`](docs/results/colab_t4_gguf/) |
| Search vs baselines | GPT-2 | MacBook MPS | [`docs/results/optimizer_macbook/`](docs/results/optimizer_macbook/) |
| Search vs baselines | TinyLlama 1.1B | Colab T4 | [`docs/results/optimizer_colab_t4/`](docs/results/optimizer_colab_t4/) |

### Search vs baselines (centerpiece)

Same space, wall-clock only. HV is throughput–memory hypervolume relative to exhaustive grid.

**Mac GPT-2 / MPS** (8 configs, budget 4):

| Strategy | Evals | HV / grid |
|----------|------:|----------:|
| Grid | 8 | 1.00 |
| Random | 4 | **0.32** |
| InferLite | 4 | 0.30 |
| Heuristic | 1 | 0.22 |

On this seed random slightly beat InferLite. Neither recovered the full front. n=8 is small.

![Mac search](docs/results/optimizer_macbook/figures/search_vs_baselines.png)

**Colab TinyLlama / T4** (4 configs, budget 2):

| Strategy | Evals | HV / grid |
|----------|------:|----------:|
| Grid | 4 | 1.00 |
| InferLite | 2 | **0.85** |
| Heuristic | 1 | 0.22 |
| Random | 2 | 0.041 |

n=4; first FP16 job had `warmup_runs=0` (cold start). Do not copy these HV numbers onto the Mac table.

![T4 search](docs/results/optimizer_colab_t4/figures/search_vs_baselines.png)

### Predictor ablation

Ridge models: hardware + quantization + workload → P95, tokens/s, memory. Leave-one-out.

**Mac, n=8:** full P95 R² **0.86**, tokens/s **0.51**, memory **0.999**. Drop quantization → memory/throughput collapse (R² −1.52 / −1.26). Drop workload → P95 collapses (R² −0.72). Hardware features do nothing on one Mac.

![Mac ablation](docs/results/optimizer_macbook/figures/predictor_ablation.png)

**T4, n=4:** memory R² **0.999**; P95 and tokens/s R² negative (−6.66 / −8.44). Drop quantization → memory R² −2.92.

### What actually ran (not Llama-3)

**T4 TinyLlama, 16 new tokens** — backends labeled; memory is not comparable across llama.cpp vs Hugging Face:

| Backend | Method | tok/s | P95 e2e (ms) | Memory |
|---------|--------|------:|-------------:|--------|
| llama.cpp | GGUF Q4_K_M | **172.5** | **105.8** | 9.125 MB engine snapshot, **not** VRAM |
| Hugging Face | FP16 | 35.0 | 468 | 2108 MB CUDA |
| Hugging Face | INT4 NF4 | 15.3 | 1057 | 803 MB CUDA |

**Mac GPT-2:** FP16 211 tok/s vs FP32 146 tok/s. bitsandbytes / AWQ / GPTQ / GGUF / vLLM: **unsupported** on that Mac, not ranked last.

AWQ 71 TPS, GPTQ 79 TPS, TensorRT-LLM 295 TPS, and 4.34× speculative speedup from the old playground are **not** measurements. They are not in these tables.

## Reproducibility

```bash
cd backend
source .venv/bin/activate    # this Mac has no `python`; the venv provides it
python -m research capabilities
python -m research suite --config ../configs/macbook_cpu.yaml
python -m research optimize --config ../configs/optimizer_macbook.yaml
python -m pytest
```

Colab T4: [`notebooks/inferlite_colab.ipynb`](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb) — install **only** `backend/requirements-colab.txt`.

## Paper

[`docs/paper.md`](docs/paper.md)

## Limits

MPS memory is RSS. llama.cpp’s 9.125 MB is not llama.cpp VRAM. Search is seed-sensitive (Mac n=8; T4 n=4 with a cold-start first job). Sliding-window is truncation. Speculative / continuous batching here are research loops, not vLLM. Energy, MMLU, GSM8K, HumanEval, and 7B+ Llama/Mistral/Qwen are not scored.

```
@software{inferlite2026,
  title = {InferLite},
  year  = {2026},
  url   = {https://github.com/Shivani767/llm-inferlite}
}
```

Apache License 2.0
