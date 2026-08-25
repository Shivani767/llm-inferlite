from pathlib import Path

from research.schema import BenchmarkMetrics, ExperimentRecord, PercentileStats, Status, unsupported
from research.viz import plot_budget_sweep, plot_suite


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


def test_plot_budget_sweep_writes_hv_figure(tmp_path: Path):
    study = {
        "n_search_space": 40,
        "seeds": [42, 123, 456, 789, 1000],
        "sweep": [
            {
                "budget": 2,
                "strategies": {
                    "random": {
                        "hv_vs_grid": {
                            "mean": 0.20,
                            "std": 0.05,
                            "ci95_low": 0.14,
                            "ci95_high": 0.26,
                        }
                    },
                    "inferlite": {
                        "hv_vs_grid": {
                            "mean": 0.22,
                            "std": 0.04,
                            "ci95_low": 0.17,
                            "ci95_high": 0.27,
                        }
                    },
                    "heuristic": {
                        "hv_vs_grid": {
                            "mean": 0.10,
                            "std": 0.0,
                            "ci95_low": 0.10,
                            "ci95_high": 0.10,
                        }
                    },
                },
            },
            {
                "budget": 4,
                "strategies": {
                    "random": {
                        "hv_vs_grid": {
                            "mean": 0.35,
                            "std": 0.06,
                            "ci95_low": 0.28,
                            "ci95_high": 0.42,
                        }
                    },
                    "inferlite": {
                        "hv_vs_grid": {
                            "mean": 0.40,
                            "std": 0.05,
                            "ci95_low": 0.34,
                            "ci95_high": 0.46,
                        }
                    },
                    "heuristic": {
                        "hv_vs_grid": {
                            "mean": 0.10,
                            "std": 0.0,
                            "ci95_low": 0.10,
                            "ci95_high": 0.10,
                        }
                    },
                },
            },
        ],
    }
    paths = plot_budget_sweep(study, tmp_path)
    assert paths
    assert paths[0].name == "hv_vs_budget.png"
    assert paths[0].exists()
