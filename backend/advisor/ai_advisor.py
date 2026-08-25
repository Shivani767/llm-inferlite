from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class OptimizationRecommendation(BaseModel):
    kind: str = "heuristic_policy"
    measured: bool = False
    best_runtime: str
    best_quantization: str
    best_batch_size: int
    best_cache_strategy: str
    explanation: str
    disclaimer: str


class AIResearchAdvisor:
    def __init__(self, benchmark_history: List[Dict[str, Any]]):
        self.history = benchmark_history or []

    def get_recommendation(
        self,
        model_name: str,
        hardware: str,
        latency_sla_ms: int,
        budget_usd: float,
    ) -> OptimizationRecommendation:
        measured = [h for h in self.history if h.get("status") == "measured"]
        if measured:
            best = max(
                measured,
                key=lambda h: ((h.get("metrics") or {}).get("tokens_per_sec") or {}).get("mean") or 0,
            )
            return OptimizationRecommendation(
                kind="from_measured_history",
                measured=True,
                best_runtime=best.get("backend", "unknown"),
                best_quantization=best.get("method", "unknown"),
                best_batch_size=int((best.get("config") or {}).get("batch_size") or 1),
                best_cache_strategy="dynamic",
                explanation="Selected the measured history row with the highest tokens/sec on this tracker.",
                disclaimer="This ranking uses stored InferLite measurements only.",
            )

        hw = hardware.lower()
        if "mac" in hw or "mps" in hw or "cpu" in hw:
            runtime, quant, cache = "transformers", "fp32", "dynamic"
            text = "No measurements yet. On Mac/CPU, start with transformers fp32, then dynamic int8 and GGUF."
        elif "t4" in hw or "colab" in hw or "cuda" in hw:
            runtime, quant, cache = "transformers", "int4_bnb", "dynamic"
            text = "No measurements yet. On CUDA, measure fp16 vs bitsandbytes INT8/INT4 before choosing AWQ/GPTQ checkpoints."
        else:
            runtime, quant, cache = "transformers", "fp16", "dynamic"
            text = "No measurements yet. Run `python -m research capabilities` then a config suite on the target box."

        return OptimizationRecommendation(
            best_runtime=runtime,
            best_quantization=quant,
            best_batch_size=1,
            best_cache_strategy=cache,
            explanation=text,
            disclaimer=(
                "Heuristic only. InferLite will not invent expected TPS, cost, or memory. "
                "Populate benchmark history by running a suite."
            ),
        )
