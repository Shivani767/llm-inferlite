"""Runtime comparison: probe backends and measure, or label unsupported. No profile tables."""

from typing import List, Optional

from pydantic import BaseModel, Field

from research.engine import run_benchmark
from research.schema import Status


class RuntimeBenchmarkMetrics(BaseModel):
    runtime: str
    status: str
    reason: Optional[str] = None
    ttft_ms: Optional[float] = None
    tps: Optional[float] = None
    latency_avg_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    load_time_s: Optional[float] = None
    device: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


RUNTIME_TO_METHOD = {
    "transformers": ("transformers", "fp32"),
    "vllm": ("vllm", "vllm"),
    "onnx_runtime": ("onnx", "onnx"),
    "llama.cpp": ("llama.cpp", "gguf"),
    "tensorrt_llm": ("tensorrt_llm", "tensorrt_llm"),
}


class RuntimeBenchmarkLab:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name

    def benchmark_runtime(self, runtime: str, **kwargs) -> RuntimeBenchmarkMetrics:
        backend, method = RUNTIME_TO_METHOD.get(runtime, (runtime, runtime))
        rec = run_benchmark(
            model_id=self.model_name,
            method=method,
            backend=backend,
            max_new_tokens=int(kwargs.get("max_new_tokens", 32)),
            measure_runs=int(kwargs.get("measure_runs", 2)),
            warmup_runs=int(kwargs.get("warmup_runs", 1)),
            gguf_file=kwargs.get("gguf_file"),
            filename=kwargs.get("gguf_file"),
        )
        m = rec.metrics
        if rec.status != Status.MEASURED or m is None:
            return RuntimeBenchmarkMetrics(
                runtime=runtime,
                status=rec.status.value,
                reason=rec.reason,
                device=rec.device,
                notes=rec.notes,
            )
        mem = None
        if m.peak_gpu_allocated_mb is not None:
            mem = round(m.peak_gpu_allocated_mb / 1024.0, 4)
        elif m.peak_rss_mb is not None:
            mem = round(m.peak_rss_mb / 1024.0, 4)
        return RuntimeBenchmarkMetrics(
            runtime=runtime,
            status=rec.status.value,
            ttft_ms=m.ttft_ms.mean if m.ttft_ms else None,
            tps=m.tokens_per_sec.mean if m.tokens_per_sec else None,
            latency_avg_ms=m.e2e_latency_ms.mean if m.e2e_latency_ms else None,
            latency_p95_ms=m.e2e_latency_ms.p95 if m.e2e_latency_ms else None,
            latency_p99_ms=m.e2e_latency_ms.p99 if m.e2e_latency_ms else None,
            memory_usage_gb=mem,
            load_time_s=m.load_time_s,
            device=rec.device,
            notes=rec.notes,
        )

    def run_full_comparison(self, runtimes: Optional[List[str]] = None, **kwargs) -> List[RuntimeBenchmarkMetrics]:
        runtimes = runtimes or list(RUNTIME_TO_METHOD.keys())
        return [self.benchmark_runtime(rt, **kwargs) for rt in runtimes]

    def generate_report(self, results: List[RuntimeBenchmarkMetrics]) -> str:
        lines = [
            f"# Runtime measurements: {self.model_name}",
            "",
            "| Runtime | Status | TTFT ms | TPS | P95 ms | Reason |",
            "|---------|--------|---------|-----|--------|--------|",
        ]
        for r in results:
            lines.append(
                f"| {r.runtime} | {r.status} | {r.ttft_ms if r.ttft_ms is not None else '—'} | "
                f"{r.tps if r.tps is not None else '—'} | "
                f"{r.latency_p95_ms if r.latency_p95_ms is not None else '—'} | "
                f"{(r.reason or '')[:80]} |"
            )
        return "\n".join(lines)


class InferenceRuntime:
    @staticmethod
    def values():
        return list(RUNTIME_TO_METHOD.keys())
