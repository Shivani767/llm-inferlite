# Colab Tesla T4 / llama.cpp GGUF (TinyLlama 1.1B Q4_K_M)

Fill this folder from `notebooks/inferlite_colab.ipynb` **llama.cpp** cells after a prebuilt `llama-cpp-python` wheel is installed. Do not compile from source on Colab.

- Config: `configs/colab_t4_gguf.yaml`
- Checkpoint: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` / `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (~670 MB)
- Workload: same prompt and 16 new tokens as the T4 lite suite, so tok/s is comparable to FP16 / bitsandbytes INT4
- If `n_gpu_layers=0`, the row is a **CPU** llama.cpp measurement — label it that way, do not mix with CUDA offload

Paste notebook stdout here. Do not invent tok/s.
