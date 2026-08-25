from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from research.quality import compute_perplexity, WIKI_SNIPPET


class EvaluationResult(BaseModel):
    benchmark_name: str
    status: str
    score: Optional[float] = None
    accuracy: Optional[float] = None
    samples: Optional[int] = None
    latency_avg_ms: Optional[float] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ResearchEvaluator:
    """
    Quality evaluation. MMLU/GSM8K/HumanEval are not silently simulated.
    Perplexity is computed only when a live model/tokenizer is provided.
    """

    def __init__(self, model_id: str, version_tag: str, model=None, tokenizer=None, device: str = "cpu"):
        self.model_id = model_id
        self.version_tag = version_tag
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    async def run_mmlu(self, subjects: Optional[List[str]] = None) -> EvaluationResult:
        return EvaluationResult(
            benchmark_name="MMLU",
            status="unsupported",
            reason="MMLU is not executed in InferLite. Wire a dataset runner yourself; scores are never invented.",
            metadata={"subjects": subjects or ["all"]},
        )

    async def run_gsm8k(self) -> EvaluationResult:
        return EvaluationResult(
            benchmark_name="GSM8K",
            status="unsupported",
            reason="GSM8K is not executed in InferLite. Scores are never invented.",
        )

    async def run_humaneval(self) -> EvaluationResult:
        return EvaluationResult(
            benchmark_name="HumanEval",
            status="unsupported",
            reason="HumanEval is not executed in InferLite. Scores are never invented.",
        )

    async def run_perplexity(self) -> EvaluationResult:
        if self.model is None or self.tokenizer is None:
            return EvaluationResult(
                benchmark_name="perplexity_builtin_passage",
                status="unsupported",
                reason="Provide a loaded model and tokenizer to measure perplexity.",
            )
        ppl = compute_perplexity(self.model, self.tokenizer, self.device, WIKI_SNIPPET)
        if ppl is None:
            return EvaluationResult(
                benchmark_name="perplexity_builtin_passage",
                status="error",
                reason="perplexity computation failed",
            )
        return EvaluationResult(
            benchmark_name="perplexity_builtin_passage",
            status="measured",
            score=ppl,
            samples=1,
            metadata={"passage": "builtin English snippet, not WikiText-2 full split"},
        )

    def compare_to_base(self, quantized_results: EvaluationResult, base_results: EvaluationResult) -> Dict[str, Any]:
        if quantized_results.status != "measured" or base_results.status != "measured":
            return {
                "status": "unsupported",
                "reason": "both sides must be measured",
            }
        if not quantized_results.score or not base_results.score:
            return {"status": "unsupported", "reason": "missing scores"}
        return {
            "benchmark": quantized_results.benchmark_name,
            "score_delta": quantized_results.score - base_results.score,
            "status": "measured",
        }
