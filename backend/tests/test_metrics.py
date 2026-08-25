from research.metrics import percentile_stats, tokens_per_second, mean_std_ci95
from research.schema import ExperimentRecord, Status, unsupported


def test_percentile_stats_known_values():
    stats = percentile_stats([10, 20, 30, 40, 50])
    assert stats is not None
    assert stats.n == 5
    assert stats.min == 10
    assert stats.max == 50
    assert stats.p50 == 30
    assert stats.ci95_low is not None
    assert stats.ci95_high is not None
    assert stats.ci95_low < stats.mean < stats.ci95_high


def test_percentile_stats_empty():
    assert percentile_stats([]) is None
    assert percentile_stats([float("nan")]) is None


def test_tokens_per_second():
    assert tokens_per_second(10, 2.0) == 5.0
    assert tokens_per_second(0, 1.0) is None
    assert tokens_per_second(10, 0) is None


def test_mean_std_ci95_five_seeds():
    stats = mean_std_ci95([0.30, 0.32, 0.28, 0.31, 0.29])
    assert stats["n"] == 5
    assert abs(stats["mean"] - 0.30) < 1e-9
    assert stats["std"] is not None and stats["std"] > 0
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_mean_std_ci95_empty_is_null():
    stats = mean_std_ci95([None, float("nan")])
    assert stats["n"] == 0
    assert stats["mean"] is None
    assert stats["ci95_low"] is None


def test_unsupported_forbids_metrics():
    rec = unsupported(
        experiment_id="x",
        experiment_type="quantization",
        model_id="m",
        backend="transformers",
        method="awq",
        device="cpu",
        precision="awq",
        reason="no CUDA",
        environment={},
    )
    assert rec.status == Status.UNSUPPORTED
    assert rec.metrics is None
    dumped = rec.model_dump()
    assert dumped["metrics"] is None
    assert "no CUDA" in dumped["reason"]
