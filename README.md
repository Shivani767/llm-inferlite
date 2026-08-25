# InferLite

**A measurement-first research toolkit for LLM inference on a laptop and free Colab GPUs.**

InferLite records wall-clock inference metrics on the machine you actually have. It does not interpolate TensorRT-LLM, vLLM, AWQ, or MMLU numbers for hardware that is not present.

> Status vocabulary: **measured** (timings from this run) · **unsupported** (cannot run here; reason recorded; metrics left null) · **error** (attempted and failed). Unsupported rows are never scored.

This README reports a completed study on **GPT-2 / DistilGPT-2**, Apple Silicon **MPS**, 2026-08-25. It is not a Llama-3-8B result.

---

## Abstract

Inference papers often mix kernel-level claims (AWQ, GPTQ, PagedAttention, speculative decoding) with numbers collected on datacenter GPUs. Practitioners on a MacBook or a free T4 cannot reproduce those tables, and many public “research platforms” fill the gap with simulated profiles.

InferLite asks a narrower question: **on this device, which inference techniques can be executed, and what are the measured trade-offs among those that can?**

On a MacBook Air (arm64, PyTorch 2.13, MPS) running `configs/macbook_cpu.yaml`:

- **19 / 29** experiments were **measured**; **10** were **unsupported**; **0** errors.
- **FP16 vs FP32** on Hugging Face GPT-2: **+45% tokens/s** (211 vs 146) and **−32% P95 e2e latency** (118 vs 175 ms).
- **PyTorch dynamic INT8, bitsandbytes INT8/INT4, AWQ, GPTQ, GGUF, vLLM PagedAttention, SmoothQuant, SqueezeLLM, BF16** did not run. They are listed with reasons, not guessed TPS.
- **Greedy speculative decoding** (GPT-2 target, DistilGPT-2 draft) accepted **68–75%** of draft tokens but was **slower** than autoregressive decode (0.47–0.60×). High acceptance is not sufficient when the verify loop is a Python/eager implementation on MPS.
- **KV cache on** (`use_cache=True`) outperforms recompute (`no_cache`) on end-to-end decode. A “sliding window” point appears on the Pareto front only because the prompt is **truncated**; it is not Mistral sliding-window attention.

Raw CSV, JSON, and figures: [`docs/results/macbook_mps_gpt2/`](docs/results/macbook_mps_gpt2/).

---

## Research questions

1. Which quantization and runtime methods are *executable* on Apple MPS versus CUDA Colab, without silent fallbacks?
2. For executable methods, what are TTFT, tokens/s, P50/P95/P99, RSS, and load time under a fixed protocol?
3. Does KV-cache reuse, greedy speculative decoding, or in-process continuous batching improve wall-clock decode on this stack?
4. What is the Pareto front among **measured** (latency, memory, throughput) points only?

---

## Contributions

- A **capability matrix** that probes libraries and devices before timing (`python -m research capabilities`).
- A **benchmark engine** with a prefill/decode loop: TTFT = first forward; inter-token times = subsequent steps; tokens/s = completion tokens / wall time.
- Experiment modules for quantization, KV-cache scaling, greedy speculative verification, static vs continuous batching, and Pareto filtering.
- **JSON + CSV** records that include environment metadata (OS, CPU, Torch, CUDA/MPS, package versions, seed).
- A **Colab notebook** (`notebooks/inferlite_colab.ipynb`) for CUDA-only methods that a MacBook cannot run.
- An honesty rule: if a method is missing, InferLite writes `unsupported` and leaves metrics empty.

---

## Experimental protocol

| Item | Setting (this study) |
|------|----------------------|
| Config | `configs/macbook_cpu.yaml` |
| Model | `gpt2` (124M); draft `distilgpt2` |
| Device | Apple MPS (`cuda=false`) |
| Software | Python 3.12.13, PyTorch 2.13.0 (MPS build), Transformers 5.15.1, Accelerate 1.14.0 |
| Seed | 42; greedy decode (`temperature=0`) |
| Warmup / measure | 1 warmup, 3 timed runs (KV/spec/batching use 2) |
| Prompt | `"The future of efficient language model inference is"` |
| New tokens | 24 (quantization / speculative); 12 (KV); 8 (batching) |
| Timing | `time.perf_counter` |
| Memory | process RSS (MPS has no CUDA allocated-byte API) |
| Pareto | minimize P95 e2e latency and RSS; maximize tokens/s; **measured rows only** |

Reproduction:

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m research capabilities
python -m research suite --config ../configs/macbook_cpu.yaml
```

---

## Results (MacBook MPS, 2026-08-25)

### Quantization / precision

Only methods that loaded and generated are scored.

| Method | Status | Tokens/s (mean) | P95 e2e (ms) | TTFT mean (ms) | Peak RSS (MB) | Weight (MB) |
|--------|--------|-----------------|--------------|----------------|---------------|-------------|
| FP32 | measured | 145.8 | 174.8 | 2.16 | 761 | 475 |
| FP16 | measured | 211.2 | 118.3 | 2.43 | 1064 | 237 |
| BF16 | unsupported | — | — | — | — | CUDA required |
| Dynamic INT8 | unsupported | — | — | — | — | `quantized::linear_prepack` engine missing on this PyTorch/Mac build |
| bitsandbytes INT8/INT4 | unsupported | — | — | — | — | CUDA + bitsandbytes |
| AWQ / GPTQ | unsupported | — | — | — | — | CUDA + autoawq / auto-gptq or a pre-quantized checkpoint |
| GGUF Q4_K_M | unsupported | — | — | — | — | `llama-cpp-python` not installed |
| SmoothQuant / SqueezeLLM | unsupported | — | — | — | — | not bundled; no fallback scores |

FP16 is the only measured compression of the dense GPT-2 weights on MPS in this run (halved `model_weight_mb`). RSS rose versus FP32; treat RSS as process footprint, not a CUDA allocator curve.

![Precision: throughput and P95 latency](docs/results/macbook_mps_gpt2/figures/quantization_comparison.png)

![Methods that were not scored](docs/results/macbook_mps_gpt2/figures/unsupported_experiments.png)

### KV-cache scaling (context 32 / 64 / 128)

| Strategy | What we actually ran |
|----------|----------------------|
| `dynamic` | Hugging Face `use_cache=True` |
| `no_cache` | recompute full prefix each step |
| `sliding_window` | **prompt truncation** to a fixed window (not a fused SWA kernel) |
| `prefix` | second forward that consumes `past_key_values` |
| `paged_attention` | unsupported (vLLM not present) |

With cache enabled, e2e latency stays in a band as context grows 32→128 (dynamic P95 298→337 ms). Disabling the cache inflates decode time (no_cache P95 up to ~1.0 s at 64 tokens). Sliding-window rows look fast because the **effective prompt is shorter**, so they are not comparable to full-context `dynamic`.

![KV memory and TTFT vs context length](docs/results/macbook_mps_gpt2/figures/kv_cache_scaling.png)

### Speculative decoding

Greedy draft–target verification, shared GPT-2 tokenizer family.

| Method | Tokens/s | Draft acceptance | Speedup vs measured baseline |
|--------|----------|------------------|------------------------------|
| Baseline (target only) | 164.0 | — | 1.00× |
| γ = 2 | 77.4 | 0.75 | **0.47×** |
| γ = 4 | 98.2 | 0.68 | **0.60×** |

Acceptance is healthy; wall-clock speedup is not. On MPS, two eager models plus a Python verify loop cost more than they save at these sequence lengths. That is a measurement, not a claim that speculative decoding “does not work” in vLLM/TRT-LLM.

![Speculative throughput and acceptance](docs/results/macbook_mps_gpt2/figures/speculative_decoding.png)

### Continuous batching

In-process scheduler versus `model.generate()` static batch (4 requests, max batch 2, 8 new tokens). Static batch: 53.1 tok/s wall-clock; continuous: 41.4 tok/s. This is **not** vLLM continuous batching. Decoder-only right-padding was flagged by Transformers; timings remain wall-clock but generation quality for padded batches should use `padding_side='left'` in a follow-up.

### Pareto front (measured only)

Objectives: min P95 latency, min RSS, max tokens/s.

| Point | Why it sits on the front |
|-------|--------------------------|
| FP32 | lowest RSS among scored dense runs |
| FP16 | higher tokens/s than FP32, lower P95 |
| `sliding_window` @ 64 | highest tokens/s — **excluded from any “best KV method” claim**; truncated prompt |

`no_cache` and full-context `dynamic` points are dominated on this plot. Do not report sliding-window as a production KV optimizer from this figure.

![Throughput–memory Pareto](docs/results/macbook_mps_gpt2/figures/pareto_throughput_memory.png)

---

## What this study does *not* show

- Llama-3-8B, MMLU, GSM8K, or HumanEval (those evals are labeled unsupported; scores are never invented).
- TensorRT-LLM, vLLM throughput, or datacenter GPU ranking.
- That AWQ/GPTQ/GGUF are “worse” on a Mac — they **did not run**.
- That speculative decoding cannot win with fused kernels or a cheaper draft.

CUDA follow-up: `notebooks/inferlite_colab.ipynb` + `configs/colab_t4.yaml` (TinyLlama 1.1B). Optional Mac GGUF: `pip install llama-cpp-python` and set `gguf_repo` / `gguf_file`.

---

## Architecture

```
CLI / Colab / FastAPI
        │
        ▼
 research.runner  ← YAML/JSON configs
        │
        ├── engine          TTFT, TPS, P50/P95/P99, load, RSS/CUDA memory, env
        ├── quantization    FP32/FP16/BF16, dynamic INT8, bnb, AWQ, GPTQ, GGUF
        ├── kv_cache        context sweep, prefix reuse, truncated window
        ├── speculative     greedy draft–target verification
        ├── batching        static generate() vs in-process continuous batch
        └── pareto          measured rows only
        │
        ▼
 JSON + CSV                 PNG/PDF figures
```

The FastAPI registry, ONNX lab, and **serving-queue simulator** remain. The simulator is Poisson-arrival capacity planning, not a model benchmark.

| Path | Role |
|------|------|
| `backend/research/` | measurement core |
| `configs/` | experiment YAML |
| `notebooks/inferlite_colab.ipynb` | CUDA / Colab T4 |
| `docs/results/macbook_mps_gpt2/` | this study’s artifacts |
| `backend/tests/` | offline tiny-GPT-2 tests (no Hub download) |

```bash
python -m research env
python -m research bench --model gpt2 --method fp16
python -m pytest   # from backend/
```

---

## Limitations

- Apple MPS memory is RSS, not `torch.cuda.max_memory_allocated`.
- Dynamic INT8 is `torch.quantization.quantize_dynamic` on `nn.Linear`, and it failed on this Mac build.
- Sliding-window KV is prompt truncation.
- Continuous batching is a research scheduler, not vLLM.
- Speculative decoding is greedy and Python-side.
- AWQ/GPTQ *conversion* is not a one-click calibrator; loading a pre-quantized checkpoint is attempted when CUDA libraries exist.
- Free Colab T4 is a different device class; do not pool those numbers with this MPS table without labeling both environments.

---

## Citation

If you use InferLite or these numbers, cite the repository and the environment row (device, PyTorch, model, date). Do not cite this README as a Llama-3-8B or TensorRT-LLM result.

```
@software{inferlite2026,
  title  = {InferLite: measurement-first LLM inference research},
  year   = {2026},
  url    = {https://github.com/Shivani767/llm-inferlite}
}
```

## License

Apache License 2.0
