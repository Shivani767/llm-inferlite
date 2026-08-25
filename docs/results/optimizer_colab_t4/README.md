# Colab Tesla T4 search study (TinyLlama 1.1B)

After a Colab **RAM crash**, do not re-run the lite suite and search in one session. Delete the runtime, run setup cells, then search only (`configs/optimizer_colab_t4.yaml` is two unique TinyLlama loads).

- Config: `configs/optimizer_colab_t4.yaml`
- Space: fp16 / INT4, context 32, 8 new tokens (**2 unique loads**; strategies share a cache)
- InferLite / random budget: 2
- Do not paste Mac GPT-2 search numbers here

After the Colab run, save `experiments.csv`, `search_study.json` comparison, ablation, and `figures/` from `/content/llm-inferlite/backend/results/optimizer_colab_t4/`.
