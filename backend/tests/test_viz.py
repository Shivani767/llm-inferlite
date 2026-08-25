from pathlib import Path

from research.schema import BenchmarkMetrics, ExperimentRecord, PercentileStats, Status, unsupported
from research.viz import plot_suite


def test_plot_suite_writes_figures(tmp_path: Path):
    measured = ExperimentRecord(
        experiment_id="m1",
        experiment_type="quantization",
        status=Status.MEASURED,
        timestamp_utc="2026-01-01T00:00:00Z",
        model_id="tiny",
        backend="transformers",
        method="fp32",
        device="cpu",
        precision="fp32",
        environment={"machine": "test", "torch": {}},
        metrics=BenchmarkMetrics(
            e2e_latency_ms=PercentileStats(
                n=2, mean=20, std=0, min=20, p50=20, p95=22, p99=22, max=22
            ),
            tokens_per_sec=PercentileStats(
                n=2, mean=8, std=0, min=8, p50=8, p95=8, p99=8, max=8
            ),
            peak_rss_mb=200.0,
        ),
    )
    skip = unsupported(
        experiment_id="u1",
        experiment_type="quantization",
        model_id="tiny",
        backend="transformers",
        method="awq",
        device="cpu",
        precision="awq",
        reason="CUDA required",
        environment={"machine": "test"},
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    paths = plot_suite([measured, skip], tmp_path)
    names = {p.name for p in paths}
    assert "quantization_comparison.png" in names
    assert "unsupported_experiments.png" in names
    assert "pareto_throughput_memory.png" in names
