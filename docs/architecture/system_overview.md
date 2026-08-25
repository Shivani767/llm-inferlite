# InferLite architecture

Measurement-first inference research: capability probing, wall-clock metrics, explicit `unsupported` labels, multi-objective search, and a ridge predictor that is only fit on measured rows.

## Research data plane

```
configs/*.yaml
  → research.runner
      → engine (TTFT, TPS, percentiles, CI, load, memory)
      → search (grid / random / heuristic / InferLite)
      → predictor + ablation
      → JSON/CSV + figures
```

**Simulation is separate:** `services.simulator` / `python -m research simulate` is a Poisson queueing model. It is not an LLM benchmark.

## Published studies

- MacBook MPS GPT-2 measurement suite: [`docs/results/macbook_mps_gpt2/`](../results/macbook_mps_gpt2/)
- Colab T4 TinyLlama lite: [`docs/results/colab_t4_lite/`](../results/colab_t4_lite/)
- MPS search vs baselines: [`docs/results/optimizer_macbook/`](../results/optimizer_macbook/)
- Paper: [`docs/paper.md`](../paper.md)

## Honesty rule

Missing library, GPU, or checkpoint → `status=unsupported`, null metrics, recorded reason. Predicted values are tagged as predictions and are never written as `status=measured`.
