# InferLite

Wall-clock LLM inference measurements on the hardware you have — a MacBook (Apple MPS) and free Colab GPUs. Methods that cannot run are recorded as **unsupported** with a reason. Metrics are left empty. Nothing is simulated.

This page reports two completed runs. They are different models and devices. Do not stack the tables.

| Study | Model | Device | Date | Artifacts |
|-------|-------|--------|------|-----------|
| MacBook MPS | GPT-2 / DistilGPT-2 | Apple Silicon MPS | 2026-08-25 | [`docs/results/macbook_mps_gpt2/`](docs/results/macbook_mps_gpt2/) |
| Colab T4 lite | TinyLlama 1.1B Chat | Tesla T4 CUDA | 2026-08-25 | [`docs/results/colab_t4_lite/`](docs/results/colab_t4_lite/) |

**measured** — timed on that machine · **unsupported** — did not run; not scored · **error** — attempted and failed

Colab notebook: [`notebooks/inferlite_colab.ipynb`](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb)

---

## Logic

Published inference tables often assume CUDA kernels (AWQ, GPTQ, bitsandbytes, vLLM, TensorRT-LLM). Those stacks are not present on a typical MacBook. InferLite therefore **probes the machine first**, then times only what loads.

Each timed generation uses a prefill/decode loop:

- **TTFT** — wall time of the first forward (prefill → first token)
- **Inter-token latency** — each later decode step
- **Tokens/s** — completion tokens / end-to-end wall time
- **P50 / P95 / P99** — percentiles over measured runs
- **Memory** — process RSS on MPS; CUDA `max_allocated` on T4
- **Load time** — `from_pretrained` through a ready model

Decode is greedy (`temperature=0`), seed 42, warmup discarded, then N timed runs. Environment (OS, CPU, PyTorch, MPS/CUDA, package versions) is stored with every record.

Pareto uses **measured** rows only: minimize P95 latency and RSS, maximize tokens/s. Unsupported rows never enter the front.

---

## Setup (MacBook MPS run)

| | |
|--|--|
| Config | `configs/macbook_cpu.yaml` |
| Model | `gpt2` (124M); draft `distilgpt2` |
| Device | Apple MPS (`cuda=false`) |
| Stack | Python 3.12.13, PyTorch 2.13.0 (MPS), Transformers 5.15.1, Accelerate 1.14.0 |
| Prompt | `"The future of efficient language model inference is"` |
| New tokens | 24 (precision / speculative); 12 (KV); 8 (batching) |
| Warmup / repeats | 1 / 3 (2 for KV, speculative, batching) |

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m research suite --config ../configs/macbook_cpu.yaml
```

---

## Results (MacBook MPS / GPT-2)

**29 records: 19 measured, 10 unsupported, 0 errors.**

### Precision

On MPS, the only dense GPT-2 variants that generated were FP32 and FP16. FP16 halved stored weights (475 → 237 MB), raised throughput **146 → 211 tok/s** (+45%), and cut P95 e2e **175 → 118 ms** (−32%). RSS went **up** (761 → 1064 MB); that is process footprint, not a CUDA `max_memory_allocated` curve.

| Method | Status | tok/s | P95 e2e (ms) | TTFT (ms) | RSS (MB) | Weights (MB) |
|--------|--------|------:|-------------:|----------:|---------:|-------------:|
| FP32 | measured | 145.8 | 174.8 | 2.16 | 761 | 475 |
| FP16 | measured | 211.2 | 118.3 | 2.43 | 1064 | 237 |
| BF16 | unsupported | — | — | — | — | needs CUDA |
| Dynamic INT8 | unsupported | — | — | — | — | no `linear_prepack` engine on this Mac PyTorch build |
| bitsandbytes INT8 / INT4 | unsupported | — | — | — | — | CUDA + bitsandbytes |
| AWQ / GPTQ | unsupported | — | — | — | — | CUDA + autoawq / auto-gptq (or a pre-quantized checkpoint) |
| GGUF Q4_K_M | unsupported | — | — | — | — | `llama-cpp-python` not installed |
| SmoothQuant / SqueezeLLM | unsupported | — | — | — | — | not implemented; no fallback numbers |

INT8/INT4/AWQ/GPTQ/GGUF are **absent from the comparison**, not ranked last. Ranking them would require a CUDA (or llama.cpp) run.

![FP32 vs FP16 throughput and P95](docs/results/macbook_mps_gpt2/figures/quantization_comparison.png)

![Unsupported methods (not scored)](docs/results/macbook_mps_gpt2/figures/unsupported_experiments.png)

### KV cache (context 32, 64, 128)

| Name in the logs | Implementation |
|------------------|----------------|
| `dynamic` | Hugging Face `use_cache=True` |
| `no_cache` | recompute the prefix every step |
| `sliding_window` | **truncate the prompt** (not fused sliding-window attention) |
| `prefix` | second forward using `past_key_values` |
| `paged_attention` | unsupported — vLLM not installed |

With the cache on, decode stays in a band as context grows (dynamic P95 298 → 337 ms from 32 → 128 tokens). Turning the cache off makes decode expensive (no_cache P95 ~1.0 s at 64 tokens). Sliding-window rows are faster because the **prompt is shorter**, so they are not a KV-algorithm win against full-context `dynamic`.

![KV RSS and TTFT vs context](docs/results/macbook_mps_gpt2/figures/kv_cache_scaling.png)

### Speculative decoding

Greedy draft–target verification, same tokenizer family (GPT-2 / DistilGPT-2).

| | tok/s | Draft accepted | vs baseline |
|--|------:|---------------:|------------:|
| Target only | 164.0 | — | 1.00× |
| γ = 2 | 77.4 | 0.75 | 0.47× |
| γ = 4 | 98.2 | 0.68 | 0.60× |

Draft tokens matched the target often. Wall-clock still lost: two eager models and a Python verify loop on MPS cost more than they save at 24 new tokens. That is this implementation on this device — not a statement about fused speculative engines.

![Speculative tok/s and acceptance](docs/results/macbook_mps_gpt2/figures/speculative_decoding.png)

### Batching

Four requests, max batch 2, 8 new tokens. `model.generate()` static batch: **53.1 tok/s**. In-process continuous scheduler: **41.4 tok/s**. This is not vLLM continuous batching. Transformers warned about right-padding on a decoder-only model; clocks are still valid, but a later run should set `padding_side='left'`.

### Pareto (measured rows only)

FP32 (lowest RSS), FP16 (better tok/s and P95), and sliding-window @ 64 (highest tok/s). The last point is **not** a recommended KV strategy — truncated prompt. `no_cache` and full-context `dynamic` are dominated on this plot.

![Throughput–memory Pareto](docs/results/macbook_mps_gpt2/figures/pareto_throughput_memory.png)

---

## Colab Tesla T4 / TinyLlama 1.1B

Lite quantization pass only (`configs/colab_t4_lite.yaml`). 16 new tokens, 1 warmup, 2 timed runs. Install **only** `backend/requirements-colab.txt` — never `requirements.txt` on Colab (it reinstalls Torch/ONNX and fills the disk).

| | |
|--|--|
| Config | `configs/colab_t4_lite.yaml` |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Device | Tesla T4 (14.9 GB), CUDA, Google Colab |
| Prompt | `"The future of efficient language model inference is"` |
| New tokens | 16 |
| Warmup / repeats | 1 / 2 |
| Notebook | [inferlite_colab.ipynb](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb) |

**8 records: 4 measured, 4 unsupported, 0 errors.**

The engine JSON stayed on the Colab VM. The two rows below are copied from the notebook's Pareto stdout. INT8 is measured but not on that front.

### Precision (cite these)

On this T4, **FP16 is the throughput/latency win**. INT4 (bitsandbytes NF4) is the **memory** win: GPU footprint 2108 → 803 MB, at about **0.44×** FP16 tok/s.

| Method | Status | tok/s | P95 e2e (ms) | GPU mem (MB) | Notes |
|--------|--------|------:|-------------:|-------------:|-------|
| FP16 | measured | 35.0 | 468 | 2108 | Pareto (speed) |
| INT4 NF4 (bitsandbytes) | measured | 15.3 | 1057 | 803 | Pareto (memory) |
| INT8 (bitsandbytes) | measured | ~3 | ~5400 | dominated | From the published bar chart, not engine CSV. `MatMul8bitLt` cast bfloat16→float16. |
| AWQ | unsupported | — | — | — | `autoawq` not installed |
| GPTQ | not scored as GPTQ | — | — | — | Run loaded **dense** TinyLlama; no GPTQ checkpoint was configured |
| GGUF Q4_K_M | unsupported | — | — | — | `llama_cpp` not installed |
| SmoothQuant / SqueezeLLM | unsupported | — | — | — | not implemented |

INT8 on this T4 is a **slowdown**, not a compression win. GPTQ clocks from that session are omitted on purpose.

![FP16 vs INT8 vs INT4 throughput and P95](docs/results/colab_t4_lite/figures/quantization_comparison.png)

![Unsupported methods (not scored)](docs/results/colab_t4_lite/figures/unsupported_experiments.png)

### Pareto (measured rows only)

FP16 (highest tok/s) and INT4 (lowest GPU memory). INT8 sits below both. The plot also shows a `gptq` point near FP16's memory; treat it as a dense-load artifact, not GPTQ.

![Throughput–memory Pareto](docs/results/colab_t4_lite/figures/pareto_throughput_memory.png)

KV / speculative / batching were **not** in this lite pass. Use `configs/colab_t4.yaml` only after the lite suite fits on disk.

---

## Reading the numbers

On this MacBook, the working dense path is **Hugging Face + MPS**, and **FP16 is the measured win over FP32**. Kernel quantization and vLLM did not execute, so they have no TPS there. Speculative decoding showed that **acceptance rate ≠ speedup** when the verify path is unfused. KV cache vs recompute behaves as expected; do not cite sliding-window from that suite as production PagedAttention.

On the Colab T4, **FP16 beats bitsandbytes INT8 and INT4 on tok/s and P95**; INT4 is the measured memory reduction. Do not mix TinyLlama/T4 numbers with GPT-2/MPS numbers. After a Colab disk crash: Runtime → Disconnect and delete runtime. Optional GGUF on Mac: `pip install llama-cpp-python` and set `gguf_repo` / `gguf_file`.

---

## Code

```
configs/*.yaml  →  research.runner  →  engine + experiments  →  JSON/CSV + figures
```

The engine records TTFT, tokens/s, percentiles, load time, RSS/CUDA memory, and environment. Experiments: precision/quantization, KV sweep, greedy speculative verify, static vs continuous batch, Pareto. FastAPI, ONNX lab, and a **queueing simulator** (capacity planning, not a model benchmark) are still in the tree.

```bash
python -m research capabilities
python -m research bench --model gpt2 --method fp16
cd backend && python -m pytest   # in-memory tiny GPT-2, no Hub download
```

---

## Limits of these runs

MPS memory is RSS. Dynamic INT8 failed on that Mac PyTorch build. Sliding-window is truncation. Continuous batching and speculative decode are research loops, not vLLM/TRT-LLM. AWQ/GPTQ conversion is not a calibrator here. The Colab GPTQ bar was a dense TinyLlama load and is not cited. MMLU / GSM8K / HumanEval were not run and are not invented.

```
@software{inferlite2026,
  title = {InferLite},
  year  = {2026},
  url   = {https://github.com/Shivani767/llm-inferlite}
}
```

Apache License 2.0
