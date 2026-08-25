# Colab Tesla T4 search study (TinyLlama 1.1B)

**Not published yet.** Fill from notebook stdout after a **clean** T4 runtime (skip lite + GGUF; those are already in `colab_t4_lite/` and `colab_t4_gguf/`).

- Config: `configs/optimizer_colab_t4.yaml`
- Space: fp16 / INT4 × context 32 / 64 × 8 new tokens (**4 unique loads**; strategies share a cache)
- InferLite / random budget: 2 vs grid 4
- Do not paste Mac GPT-2 search numbers here

After the Colab run, paste the printed `comparison` JSON and grid rows. Engine CSV lives at `/content/llm-inferlite/backend/results/optimizer_colab_t4/`.
