"""Speculative decoding API wrapper around greedy draft-target measurements."""

from typing import List, Optional

from pydantic import BaseModel, Field

from research.experiments.speculative import run_speculative_suite
from research.schema import Status


class SpeculativeDecodingMetrics(BaseModel):
    draft_model: str
    target_model: str
    status: str
    reason: Optional[str] = None
    acceptance_rate: Optional[float] = None
    tokens_per_second: Optional[float] = None
    speedup_over_baseline: Optional[float] = None
    latency_avg_ms: Optional[float] = None
    num_speculative_tokens: Optional[int] = None
    notes: List[str] = Field(default_factory=list)


class SpeculativeDecodingEngine:
    def __init__(self, target_model: str = "gpt2", draft_model: str = "distilgpt2"):
        self.target_model = target_model
        self.draft_model = draft_model

    def evaluate(self, num_speculative_tokens: int = 4, **kwargs) -> SpeculativeDecodingMetrics:
        recs = run_speculative_suite(
            self.target_model,
            self.draft_model,
            gammas=[num_speculative_tokens],
            max_new_tokens=int(kwargs.get("max_new_tokens", 32)),
            measure_runs=int(kwargs.get("measure_runs", 2)),
        )
        spec = next((r for r in recs if r.method.startswith("speculative")), recs[-1] if recs else None)
        if spec is None:
            return SpeculativeDecodingMetrics(
                draft_model=self.draft_model,
                target_model=self.target_model,
                status="error",
                reason="no records",
            )
        extra = (spec.metrics.extra if spec.metrics else None) or {}
        if spec.status != Status.MEASURED or spec.metrics is None:
            return SpeculativeDecodingMetrics(
                draft_model=self.draft_model,
                target_model=self.target_model,
                status=spec.status.value,
                reason=spec.reason,
                num_speculative_tokens=num_speculative_tokens,
                notes=spec.notes,
            )
        return SpeculativeDecodingMetrics(
            draft_model=self.draft_model,
            target_model=self.target_model,
            status=spec.status.value,
            acceptance_rate=extra.get("acceptance_rate_mean"),
            tokens_per_second=spec.metrics.tokens_per_sec.mean if spec.metrics.tokens_per_sec else None,
            speedup_over_baseline=extra.get("speedup_over_baseline"),
            latency_avg_ms=spec.metrics.e2e_latency_ms.mean if spec.metrics.e2e_latency_ms else None,
            num_speculative_tokens=num_speculative_tokens,
            notes=spec.notes,
        )

    def generate_report(self, **kwargs) -> str:
        m = self.evaluate(**kwargs)
        return (
            f"# Speculative decoding: {self.draft_model} + {self.target_model}\n\n"
            f"- status: {m.status}\n"
            f"- acceptance_rate: {m.acceptance_rate}\n"
            f"- tokens_per_second: {m.tokens_per_second}\n"
            f"- speedup_over_baseline: {m.speedup_over_baseline}\n"
            f"- reason: {m.reason or ''}\n"
        )
