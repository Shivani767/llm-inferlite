from pathlib import Path

from research.schema import BenchmarkMetrics, ExperimentRecord, PercentileStats, Status, unsupported
from research.storage import ResultStore


def test_json_csv_roundtrip(tmp_path: Path):
    store = ResultStore(tmp_path)
    rec = ExperimentRecord(
        experiment_id="exp1",
        experiment_type="benchmark",
        status=Status.MEASURED,
        timestamp_utc="2026-01-01T00:00:00Z",
        model_id="tiny",
        backend="transformers",
        method="fp32",
        device="cpu",
        precision="fp32",
        metrics=BenchmarkMetrics(
            load_time_s=1.2,
            ttft_ms=PercentileStats(n=2, mean=10, std=0, min=10, p50=10, p95=10, p99=10, max=10),
            tokens_per_sec=PercentileStats(n=2, mean=5, std=0, min=5, p50=5, p95=5, p99=5, max=5),
            peak_rss_mb=100.0,
        ),
        samples=[{"e2e_ms": 20}],
    )
    path = store.save(rec)
    assert path.exists()
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].experiment_id == "exp1"
    assert loaded[0].metrics.load_time_s == 1.2
    csv_text = store.csv_path.read_text(encoding="utf-8")
    assert "exp1" in csv_text
    assert "measured" in csv_text

    skip = unsupported(
        experiment_id="exp2",
        experiment_type="quantization",
        model_id="tiny",
        backend="transformers",
        method="awq",
        device="cpu",
        precision="awq",
        reason="CUDA required",
        environment={},
        timestamp_utc="2026-01-01T00:00:01Z",
    )
    store.save(skip)
    csv_text = store.csv_path.read_text(encoding="utf-8")
    assert "CUDA required" in csv_text
    assert "awq" in csv_text
