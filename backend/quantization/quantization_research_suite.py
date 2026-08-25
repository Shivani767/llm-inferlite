"""Quantization comparison API wrapper around measured experiments."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from research.experiments.quantization import DEFAULT_METHODS, run_quantization_suite
from research.schema import ExperimentRecord, Status


class QuantizationMetrics(BaseModel):
    method: str
    status: str
    reason: Optional[str] = None
    compression_ratio: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    latency_avg_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    perplexity: Optional[float] = None
    load_time_s: Optional[float] = None
    ttft_ms: Optional[float] = None
    device: Optional[str] = None
    backend: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    record: Optional[Dict] = None


def _compression(method: str, weight_mb: Optional[float], fp32_mb: Optional[float]) -> Optional[float]:
    if weight_mb and fp32_mb and weight_mb > 0:
        return round(fp32_mb / weight_mb, 3)
    return None


class QuantizationResearchSuite:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name

    def run_full_comparison(self, methods: Optional[List[str]] = None) -> List[QuantizationMetrics]:
        records = run_quantization_suite(self.model_name, methods=methods)
        fp32_mb = None
        for rec in records:
            if rec.status == Status.MEASURED and rec.method in {"fp32", "fp16"} and rec.metrics:
                fp32_mb = rec.metrics.model_weight_mb
                if rec.method == "fp32":
                    break
        return [self._to_metrics(rec, fp32_mb) for rec in records]

    def _to_metrics(self, rec: ExperimentRecord, fp32_mb: Optional[float]) -> QuantizationMetrics:
        m = rec.metrics
        if rec.status != Status.MEASURED or m is None:
            return QuantizationMetrics(
                method=rec.method,
                status=rec.status.value,
                reason=rec.reason,
                device=rec.device,
                backend=rec.backend,
                notes=rec.notes,
                record=rec.model_dump(),
            )
        mem_gb = None
        if m.peak_gpu_allocated_mb is not None:
            mem_gb = round(m.peak_gpu_allocated_mb / 1024.0, 4)
        elif m.peak_rss_mb is not None:
            mem_gb = round(m.peak_rss_mb / 1024.0, 4)
        return QuantizationMetrics(
            method=rec.method,
            status=rec.status.value,
            compression_ratio=_compression(rec.method, m.model_weight_mb, fp32_mb),
            memory_usage_gb=mem_gb,
            latency_avg_ms=m.e2e_latency_ms.mean if m.e2e_latency_ms else None,
            latency_p95_ms=m.e2e_latency_ms.p95 if m.e2e_latency_ms else None,
            latency_p99_ms=m.e2e_latency_ms.p99 if m.e2e_latency_ms else None,
            throughput_tps=m.tokens_per_sec.mean if m.tokens_per_sec else None,
            perplexity=m.perplexity,
            load_time_s=m.load_time_s,
            ttft_ms=m.ttft_ms.mean if m.ttft_ms else None,
            device=rec.device,
            backend=rec.backend,
            notes=rec.notes,
            record=rec.model_dump(),
        )

    def generate_publishable_report(self, results: List[QuantizationMetrics]) -> str:
        lines = [
            f"# Quantization measurements: {self.model_name}",
            "",
            "Only `measured` rows are wall-clock results from this machine.",
            "Unsupported methods are listed with reasons and are not scored.",
            "",
            "| Method | Status | TTFT ms | TPS | P95 ms | Memory GB | Reason |",
            "|--------|--------|---------|-----|--------|-----------|--------|",
        ]
        for r in results:
            lines.append(
                f"| {r.method} | {r.status} | {r.ttft_ms if r.ttft_ms is not None else '—'} | "
                f"{r.throughput_tps if r.throughput_tps is not None else '—'} | "
                f"{r.latency_p95_ms if r.latency_p95_ms is not None else '—'} | "
                f"{r.memory_usage_gb if r.memory_usage_gb is not None else '—'} | "
                f"{(r.reason or '')[:80]} |"
            )
        return "\n".join(lines)

    def get_pareto_optimal(self, results: List[QuantizationMetrics]) -> List[QuantizationMetrics]:
        from research.experiments.pareto import pareto_front
        from research.schema import ExperimentRecord

        records = []
        for r in results:
            if r.record:
                records.append(ExperimentRecord.model_validate(r.record))
        front_ids = {item["experiment_id"] for item in pareto_front(records)}
        return [r for r in results if r.record and r.record.get("experiment_id") in front_ids]


# Keep enum-like names for API listings
class QuantizationMethod:
    @staticmethod
    def values() -> List[str]:
        return list(DEFAULT_METHODS)
