# Colab Tesla T4 search study (TinyLlama 1.1B)

Fill this folder by running `notebooks/inferlite_colab.ipynb` cells **T4 search vs baselines** after the lite suite.

- Config: `configs/optimizer_colab_t4.yaml`
- Space: fp16 / INT4 × context 32 / 64 × 8 new tokens (4 grid jobs)
- InferLite / random budget: 3
- Do not paste Mac GPT-2 search numbers here

After the Colab run, save `experiments.csv`, `search_study.json` comparison, ablation, and `figures/` from `/content/llm-inferlite/backend/results/optimizer_colab_t4/`.
