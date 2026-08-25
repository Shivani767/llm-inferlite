"""KV-cache API wrapper around measured context-length experiments."""

from typing import List, Optional

from pydantic import BaseModel, Field

from research.experiments.kv_cache import run_kv_cache_suite
from research.schema import Status


class KVCacheMetrics(BaseModel):
    strategy: str
    status: str
    reason: Optional[str] = None
    context_length: Optional[int] = None
    memory_usage_gb: Optional[float] = None
    latency_avg_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    ttft_ms: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class KVCacheResearchSuite:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name

    def run_full_comparison(
        self,
        strategies: Optional[List[str]] = None,
        context_length: int = 128,
    ) -> List[KVCacheMetrics]:
        recs = run_kv_cache_suite(
            self.model_name,
            context_lengths=[context_length],
            strategies=strategies,
        )
        return [self._to_metrics(r) for r in recs]

    def run_context_length_study(self, strategy: str, context_lengths: Optional[List[int]] = None):
        recs = run_kv_cache_suite(
            self.model_name,
            context_lengths=context_lengths,
            strategies=[strategy],
        )
        return [self._to_metrics(r) for r in recs]

    def _to_metrics(self, rec) -> KVCacheMetrics:
        m = rec.metrics
        if rec.status != Status.MEASURED or m is None:
            return KVCacheMetrics(
                strategy=rec.method,
                status=rec.status.value,
                reason=rec.reason,
                context_length=(rec.config or {}).get("context_length"),
                notes=rec.notes,
            )
        mem = None
        if m.peak_gpu_allocated_mb is not None:
            mem = round(m.peak_gpu_allocated_mb / 1024.0, 4)
        elif m.peak_rss_mb is not None:
            mem = round(m.peak_rss_mb / 1024.0, 4)
        return KVCacheMetrics(
            strategy=rec.method,
            status=rec.status.value,
            context_length=(rec.config or {}).get("context_length"),
            memory_usage_gb=mem,
            latency_avg_ms=m.e2e_latency_ms.mean if m.e2e_latency_ms else None,
            latency_p95_ms=m.e2e_latency_ms.p95 if m.e2e_latency_ms else None,
            throughput_tps=m.tokens_per_sec.mean if m.tokens_per_sec else None,
            ttft_ms=m.ttft_ms.mean if m.ttft_ms else None,
            notes=rec.notes,
        )

    def generate_report(self, results: List[KVCacheMetrics]) -> str:
        lines = [
            f"# KV-cache measurements: {self.model_name}",
            "",
            "| Strategy | Status | Context | TTFT ms | Memory GB | Reason |",
            "|----------|--------|---------|---------|-----------|--------|",
        ]
        for r in results:
            lines.append(
                f"| {r.strategy} | {r.status} | {r.context_length} | "
                f"{r.ttft_ms if r.ttft_ms is not None else '—'} | "
                f"{r.memory_usage_gb if r.memory_usage_gb is not None else '—'} | "
                f"{(r.reason or '')[:80]} |"
            )
        return "\n".join(lines)


class KVCacheStrategy:
    @staticmethod
    def values():
        return ["dynamic", "no_cache", "sliding_window", "prefix", "paged_attention"]
