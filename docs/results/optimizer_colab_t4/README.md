# Colab Tesla T4 search study (TinyLlama 1.1B)

- Date (UTC): 2026-08-25
- Notebook: [`notebooks/inferlite_colab.ipynb`](../../../notebooks/inferlite_colab.ipynb)
- Config: `configs/optimizer_colab_t4.yaml`
- Device: Tesla T4, Google Colab (clean runtime, 10.9 GiB available RAM)
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Space: fp16 / INT4 × context 32 / 64 × 8 new tokens (**4 measured**, 0 unsupported, 0 error)
- InferLite / random budget: 2 vs grid 4
- `warmup_runs`: 0 — the first FP16 job (`c32`) is a cold start and is much slower than the later FP16 `c64` job

Do not mix with Mac GPT-2 search numbers. Do not mix with llama.cpp GGUF clocks.

## Grid rows (notebook stdout)

| Candidate | tok/s | P95 e2e (ms) | GPU mem (MB) |
|-----------|------:|-------------:|-------------:|
| fp16 \| c32 \| n8 | 6.125 | 1306.2 | 2110 |
| fp16 \| c64 \| n8 | 31.824 | 251.4 | 2114 |
| int4_bnb \| c32 \| n8 | 18.717 | 427.4 | 804 |
| int4_bnb \| c64 \| n8 | 15.370 | 520.5 | 827 |

The 6.1 tok/s FP16-c32 row is the first generate after download/load with **no warmup**. Cite it as measured, not as steady-state FP16.

## Baselines (same space, wall-clock only)

| Strategy | Evals | HV / grid |
|----------|------:|----------:|
| Grid | 4 | 1.00 |
| InferLite | 2 | **0.85** |
| Heuristic (FP16, longer context on T4) | 1 | 0.22 |
| Random | 2 | 0.041 |

On this seed InferLite recovered most of the grid front at half the budget (it evaluated `fp16|c64` and `int4_bnb|c64`). Random drew the cold-start `fp16|c32` point. n=4 is small. The 30-config T4 scale is in [`../optimizer_colab_t4_scale/`](../optimizer_colab_t4_scale/).

## Predictor ablation (leave-one-out, n=4)

| Variant | P95 R² | Tokens/s R² | Memory R² |
|---------|-------:|------------:|----------:|
| Full | −6.66 | −8.44 | 0.999 |
| No hardware | −6.66 | −8.44 | 0.999 |
| No quantization | −1.53 | −1.48 | −2.92 |
| No workload | −2.41 | −2.94 | 0.999 |

The predictor fails to generalize with only four T4 observations. Negative R² is kept on purpose. Do not “fix” these numbers. The n=30 T4 predictor is in [`../optimizer_colab_t4_scale/`](../optimizer_colab_t4_scale/).

Memory still tracks quantization. Hardware features are constant on one T4.

Files: `comparison.json`, `ablation.json`, `experiments.csv`, `figures/`.
