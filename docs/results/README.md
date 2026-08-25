# Published measured studies

| Directory | Hardware | Model | Kind |
|-----------|----------|-------|------|
| [`macbook_mps_gpt2/`](macbook_mps_gpt2/) | Apple MPS | GPT-2 | Measurement suite |
| [`colab_t4_lite/`](colab_t4_lite/) | Colab Tesla T4 | TinyLlama 1.1B | Quantization lite |
| [`optimizer_macbook/`](optimizer_macbook/) | Apple MPS | GPT-2 | Grid vs random vs heuristic vs InferLite |
| [`optimizer_colab_t4/`](optimizer_colab_t4/) | Colab Tesla T4 | TinyLlama 1.1B | Fill from notebook search cells |
| [`colab_t4_gguf/`](colab_t4_gguf/) | Colab Tesla T4 | TinyLlama 1.1B Q4_K_M | llama.cpp GGUF (172.5 tok/s) |

Do not mix rows across tables without labeling both environments. Queueing-simulator output is not stored here.
