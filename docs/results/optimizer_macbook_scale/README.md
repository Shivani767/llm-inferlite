# MacBook MPS scale study (GPT-2, n=40)

- Date (UTC): 2026-08-25 19:41
- Config: `configs/optimizer_macbook_scale.yaml`
- Device: Apple MPS
- Model: `gpt2`
- Search space: **40** configs (fp32/fp16 × context 32/48/64/96/128 × 8/12/16/24 new tokens)
- Grid: **40 measured, 0 unsupported, 0 error**
- Seeds: 42, 123, 456, 789, 1000
- Budgets: 2, 4, 8, 16
- Protocol: exhaustive grid timed once; random / InferLite / heuristic **replay** those wall-clock records. Metrics are not invented.
- Energy: **unsupported** (no NVML)

This is the centerpiece optimizer result. The earlier 8-point pilot is in [`../optimizer_macbook/`](../optimizer_macbook/). Do not mix with Colab T4 TinyLlama numbers. T4 30-config scale is not measured yet ([`../optimizer_colab_t4_scale/`](../optimizer_colab_t4_scale/)).

## InferLite vs random (key comparison)

Mean throughput–memory hypervolume / grid, 5 seeds, 95% Student-t interval.

| Budget | InferLite mean (std) | InferLite 95% CI | Random mean (std) | Random 95% CI | InferLite mean > random? |
|-------:|---------------------:|-----------------:|------------------:|--------------:|:-------------------------|
| 2 | 0.48 (0.24) | [0.18, 0.78] | 0.31 (0.36) | [−0.14, 0.76] | yes |
| 4 | 0.86 (0.03) | [0.83, 0.90] | 0.61 (0.39) | [0.12, 1.10] | yes |
| 8 | 0.92 (0.02) | [0.90, 0.95] | 0.80 (0.38) | [0.33, 1.27] | yes |
| 16 | 0.97 (0.03) | [0.93, 1.01] | 0.97 (0.01) | [0.95, 0.99] | **no** (random 0.973 vs 0.971) |

Random’s t-interval at budget 2 dips below 0 because n=5 and the sample std is large. Hypervolume ratios cannot be negative; the interval is poorly calibrated there, which is itself evidence that random is seed-unstable at a tiny budget.

InferLite does **not** dominate random at every budget. It reaches a high fraction of grid hypervolume earlier (0.86× at 4 evaluations vs 40) and with much lower variance. At 16 evaluations (~40% of the space) both methods sit near the exhaustive front and random is slightly ahead on the mean.

Heuristic (always 1 eval, prefer FP16 on MPS): **0.18×** grid HV at every budget.

![HV vs budget](figures/hv_vs_budget.png)

## Highlight: budget = 4 (10% of the grid)

| Strategy | Evals | HV / grid (mean over 5 seeds) |
|----------|------:|------------------------------:|
| Grid | 40 | 1.00 |
| InferLite | 4 | **0.86** (std 0.03, 95% CI [0.83, 0.90]) |
| Random | 4 | 0.61 (std 0.39, 95% CI [0.12, 1.10]) |
| Heuristic | 1 | 0.18 |

On seed 42 of this 40-point space InferLite was 0.83× and random was 0.11×. That is the opposite of the n=8 pilot, where random slightly beat InferLite. Ranking is seed-sensitive on small spaces; this is why the five-seed curve is the result to cite.

## Predictor ablation (leave-one-out, n=40)

| Variant | P95 R² | Tokens/s R² | Memory R² |
|---------|-------:|------------:|----------:|
| Full | 0.71 | 0.61 | 0.999 |
| No hardware | 0.71 | 0.61 | 0.999 |
| No quantization | 0.59 | −0.17 | −0.17 |
| No workload | 0.02 | 0.64 | 0.998 |

Forty measured rows are enough for a usable ridge fit on this Mac space. Quantization still carries memory and throughput. Workload still carries P95. Hardware features are constant on one Mac. The published T4 n=4 predictor (P95 R² −6.66, tokens/s R² −8.44) is left as-is: four observations are not enough, which is a dataset limit, not a number to “fix.”

![Predictor ablation](figures/predictor_ablation.png)

Files: `experiments.csv`, `budget_sweep.json`, `hv_vs_budget.csv`, `comparison.json`, `ablation.json`, `optimizer_macbook_scale.json`, `figures/`.
