# InferLite architecture

Measurement-first inference research: capability probing, wall-clock metrics, and explicit `unsupported` labels. FastAPI remains as a control plane; it no longer emits simulated Llama-3-8B scores.

## Research data plane

```
configs/*.yaml  →  research.runner  →  experiment modules  →  JSON/CSV + figures
                                      ↘ research.engine (TTFT, TPS, percentiles, load, memory, env)
```

Published MacBook MPS study: [`docs/results/macbook_mps_gpt2/`](../results/macbook_mps_gpt2/).

## Control plane

FastAPI + optional SQLite/Postgres registry, experiment tracker, Celery/Redis, Prometheus.

## Not a model benchmark

`services.simulator` is a Poisson-arrival queueing model for capacity planning. CLI and API docs label it as simulation.

## Honesty rule

Missing library, GPU, or checkpoint → `status=unsupported`, null metrics, recorded reason.
