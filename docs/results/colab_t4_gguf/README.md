# Colab Tesla T4 / llama.cpp GGUF (TinyLlama 1.1B Q4_K_M)

- Date (UTC): 2026-08-25
- Notebook: [`notebooks/inferlite_colab.ipynb`](../../../notebooks/inferlite_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/Shivani767/llm-inferlite/blob/main/notebooks/inferlite_colab.ipynb))
- Config: `configs/colab_t4_gguf.yaml`
- Device: Tesla T4, Google Colab (same session as the T4 lite suite)
- Checkpoint: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` / `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (~669 MB)
- Runtime: `llama-cpp-python` 0.3.35 from the CUDA 12.4 prebuilt wheel (`cu124`)
- Records: **1 measured, 0 unsupported, 0 error**
- `n_gpu_layers`: **-1** (full offload requested)
- `gpu_offload_compiled`: **None** (the probe helper did not report a compile flag)

Exact engine CSV stayed on the Colab VM (`/content/llm-inferlite/backend/results/colab_t4_gguf/experiments.csv`). Published numbers are the notebook stdout.

Do not stack these clocks with Hugging Face FP16/INT4 without labeling **llama.cpp vs transformers**. Do not treat 9.125 MB as llama.cpp VRAM: that value is the engine’s PyTorch/RSS snapshot, which does not track llama.cpp’s allocator.

## Exact values (notebook stdout)

| Method | Backend | tok/s | P95 e2e (ms) | mem (MB, engine snapshot) |
|--------|---------|------:|-------------:|--------------------------:|
| gguf Q4_K_M | llama.cpp | 172.500 | 105.769 | 9.125 |

Install line from stdout: `llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`.
