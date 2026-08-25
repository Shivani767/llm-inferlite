# MacBook MPS search study (GPT-2)

- Date (UTC): 2026-08-25 17:55
- Config: `configs/optimizer_macbook.yaml`
- Device: Apple MPS
- Model: `gpt2`
- Search space: 8 configs (fp32/fp16 × context 32/96 × 8/16 new tokens)
- Grid: **8 measured, 0 unsupported, 0 error**
- InferLite / random budget: 4 evaluations
- Energy: **unsupported** (no NVML)

Cite this rerun, not the earlier scratch pass. Do not mix with Colab T4 TinyLlama numbers.

## Baselines (same space, wall-clock only)

| Strategy | Evals | Throughput–memory HV vs grid |
|----------|------:|-----------------------------:|
| Grid (exhaustive) | 8 | 1.00 |
| Random | 4 | 0.32 |
| InferLite (surrogate) | 4 | 0.30 |
| Hardware heuristic | 1 | 0.22 |

Random slightly beat InferLite at the same budget on this seed. Neither recovered the grid front. n=8 is small; do not claim the surrogate replaces exhaustive search.

## Predictor ablation (leave-one-out, n=8)

| Variant | P95 R² | Tokens/s R² | Memory R² |
|---------|-------:|------------:|----------:|
| Full | 0.86 | 0.51 | 0.9995 |
| No hardware | 0.86 | 0.51 | 0.9995 |
| No quantization | 0.84 | −1.26 | −1.52 |
| No workload | −0.72 | 0.58 | 0.983 |

Memory and throughput need the precision feature. P95 needs workload (context / new tokens). Hardware features are constant on one Mac.

Files: `experiments.csv`, `comparison.json`, `ablation.json`, `figures/`.
