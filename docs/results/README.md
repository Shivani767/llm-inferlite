# Published measured studies

| Directory | Hardware | Model | Kind |
|-----------|----------|-------|------|
| [`macbook_mps_gpt2/`](macbook_mps_gpt2/) | Apple MPS | GPT-2 | Measurement suite (19 measured, 10 unsupported, 0 error) |
| [`colab_t4_lite/`](colab_t4_lite/) | Colab Tesla T4 | TinyLlama 1.1B | Quantization lite (4 measured, 4 unsupported, 0 error) |
| [`colab_t4_gguf/`](colab_t4_gguf/) | Colab Tesla T4 | TinyLlama 1.1B Q4_K_M | llama.cpp GGUF (172.5 tok/s; 9.125 MB engine snapshot, not VRAM) |
| [`optimizer_macbook/`](optimizer_macbook/) | Apple MPS | GPT-2 | Search pilot: n=8, seed 42, budget 4 (random 0.32× vs InferLite 0.30×) |
| [`optimizer_colab_t4/`](optimizer_colab_t4/) | Colab Tesla T4 | TinyLlama 1.1B | Search pilot: n=4, seed 42, budget 2, `warmup_runs=0` (InferLite 0.85× vs random 0.041×) |
| [`optimizer_macbook_scale/`](optimizer_macbook_scale/) | Apple MPS | GPT-2 | Scale: n=40, 5 seeds, budgets 2/4/8/16 (InferLite 0.86× vs random 0.61× at budget 4) |
| [`optimizer_colab_t4_scale/`](optimizer_colab_t4_scale/) | Colab Tesla T4 | TinyLlama 1.1B | Scale: n=30, 5 seeds, budgets 2/4/8/16, `warmup_runs=1` (random 0.96× vs InferLite 0.49× at budget 4) |

Do not mix rows across tables without labeling both environments and backends. Queueing-simulator output is not stored here. The two scale studies are the optimizer results to cite; the n=8 / n=4 folders are pilots.
