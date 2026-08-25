from research.experiments.pareto import dominates, pareto_front
from research.schema import BenchmarkMetrics, ExperimentRecord, PercentileStats, Status


def _rec(eid, method, tps, p95, mem):
    return ExperimentRecord(
        experiment_id=eid,
        experiment_type="quantization",
        status=Status.MEASURED,
        timestamp_utc="2026-01-01T00:00:00Z",
        model_id="m",
        backend="transformers",
        method=method,
        device="cpu",
        precision="fp32",
        metrics=BenchmarkMetrics(
            e2e_latency_ms=PercentileStats(
                n=3, mean=p95, std=0, min=p95, p50=p95, p95=p95, p99=p95, max=p95
            ),
            tokens_per_sec=PercentileStats(
                n=3, mean=tps, std=0, min=tps, p50=tps, p95=tps, p99=tps, max=tps
            ),
            peak_rss_mb=mem,
        ),
    )


def test_dominates():
    assert dominates((1, 1, -10), (2, 2, -5))
    assert not dominates((2, 2, -5), (1, 1, -10))


def test_pareto_excludes_unsupported_and_keeps_front():
    from research.schema import unsupported

    a = _rec("a", "fp32", tps=10, p95=100, mem=800)
    b = _rec("b", "int8", tps=20, p95=80, mem=400)  # better on all
    c = _rec("c", "slow", tps=5, p95=200, mem=900)
    skip = unsupported(
        experiment_id="d",
        experiment_type="quantization",
        model_id="m",
        backend="x",
        method="awq",
        device="cpu",
        precision="awq",
        reason="no cuda",
        environment={},
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    front = pareto_front([a, b, c, skip])
    methods = {item["method"] for item in front}
    assert methods == {"int8"}
    assert all("record" in item for item in front)
