# Colab Tesla T4 study (TinyLlama 1.1B)

- Date (UTC): 2026-08-25
- Notebook: [`notebooks/inferlite_colab.ipynb`](../../../notebooks/inferlite_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb))
- Config: `configs/colab_t4_lite.yaml`
- Device: Tesla T4, 14.9 GB, CUDA, Google Colab
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Records: 8 (4 measured, 4 unsupported, 0 error)

Exact engine CSV/JSON stayed on the Colab VM (`/content/llm-inferlite/backend/results/colab_t4_lite/`). What is published here comes from the notebook stdout (Pareto JSON) and the three figures Colab saved in the `.ipynb`.

Do not mix these numbers with the MacBook MPS GPT-2 study without labeling both environments.

## Exact values (notebook stdout)

| Method | tok/s | P95 e2e (ms) | GPU memory (MB) |
|--------|------:|-------------:|----------------:|
| fp16 | 34.975 | 467.92 | 2107.525 |
| int4_bnb | 15.306 | 1056.69 | 802.726 |

These two points are the Pareto front (minimize P95 and memory, maximize tok/s).

## Figure-derived (not engine CSV)

`int8_bnb` and a bar labeled `gptq` appear on `quantization_comparison.png`. Bar heights scaled against the fp16 / int4_bnb values above:

| Method | tok/s (approx) | P95 e2e (approx) |
|--------|---------------:|-----------------:|
| int8_bnb | ~3.0 | ~5.4 s |
| gptq label | ~19.5 | ~860 ms |

The `gptq` bar loaded **dense** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` because no `gptq_model_id` was set. It is **not** a GPTQ checkpoint measurement. Do not cite those clocks as GPTQ.

Unsupported **in this lite suite**: AWQ (`autoawq` missing), GGUF (`llama_cpp` missing at the time), SmoothQuant, SqueezeLLM.

GGUF was measured later in the same Colab session after installing a CUDA 12.4 `llama-cpp-python` wheel: [`colab_t4_gguf/`](../colab_t4_gguf/).
