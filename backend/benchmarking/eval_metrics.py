import torch
from typing import List, Dict, Any, Optional


class QuantizationEvalMetrics:
    """Real perplexity. Distribution comparison is not faked."""

    @staticmethod
    def calculate_perplexity(model, tokenizer, dataset_text: str, device: str = "cpu") -> float:
        from research.quality import compute_perplexity

        ppl = compute_perplexity(model, tokenizer, device, dataset_text)
        if ppl is None:
            raise RuntimeError("perplexity could not be computed")
        return ppl

    @staticmethod
    def compare_weights_distribution(original_model, quantized_model) -> Dict[str, Any]:
        return {
            "status": "unsupported",
            "reason": "weight MSE/KL is not implemented; will not return placeholder 0.0012",
        }


class RAGEvalMetrics:
    @staticmethod
    def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> Dict[str, Any]:
        return {"status": "unsupported", "reason": "faithfulness requires an NLI/judge model; not mocked"}

    @staticmethod
    def calculate_relevancy(query: str, retrieved_contexts: List[str]) -> Dict[str, Any]:
        return {"status": "unsupported", "reason": "relevancy requires a retriever metric; not mocked"}


class LLMAsAJudge:
    @staticmethod
    def score_response(query: str, response: str, reference: str = None) -> Dict[str, Any]:
        return {
            "status": "unsupported",
            "reason": "No judge model is invoked. InferLite does not invent Likert scores.",
        }
