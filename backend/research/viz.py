"""Research figures from measured records only. Unsupported rows are listed, not plotted as scores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

from research.experiments.pareto import pareto_front
from research.schema import ExperimentRecord, Status

RecordLike = Union[ExperimentRecord, dict]


def _records(items: Iterable[RecordLike]) -> List[ExperimentRecord]:
    out: List[ExperimentRecord] = []
    for item in items:
        if isinstance(item, ExperimentRecord):
            out.append(item)
        else:
            out.append(ExperimentRecord.model_validate(item))
    return out


def _measured(records: List[ExperimentRecord]) -> List[ExperimentRecord]:
    return [r for r in records if r.status == Status.MEASURED and r.metrics is not None]


def plot_suite(
    records: Iterable[RecordLike],
    output_dir: Union[str, Path],
    title_suffix: str = "",
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recs = _records(records)
    measured = _measured(recs)
    written: List[Path] = []

    style = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
    plt.rcParams.update(style)

    hardware = ""
    if recs:
        env = recs[0].environment or {}
        torch_info = env.get("torch") or {}
        gpu = (torch_info.get("gpu") or {}).get("name")
        hardware = gpu or env.get("machine") or env.get("system") or ""
    header = f"InferLite measured results{(' — ' + title_suffix) if title_suffix else ''}"
    if hardware:
        header += f"\n{hardware}"

    # 1. Quantization / method comparison
    quant = [r for r in measured if r.experiment_type in {"quantization", "benchmark"}]
    if quant:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [r.method for r in quant]
        tps = [
            r.metrics.tokens_per_sec.mean if r.metrics.tokens_per_sec else 0 for r in quant
        ]
        p95 = [
            r.metrics.e2e_latency_ms.p95 if r.metrics.e2e_latency_ms else 0 for r in quant
        ]
        axes[0].bar(range(len(labels)), tps, color="#2c5f8a")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=35, ha="right")
        axes[0].set_ylabel("tokens / sec (mean)")
        axes[0].set_title("Throughput")
        axes[1].bar(range(len(labels)), p95, color="#8a3c2c")
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=35, ha="right")
        axes[1].set_ylabel("end-to-end P95 (ms)")
        axes[1].set_title("Latency")
        fig.suptitle(header)
        fig.tight_layout()
        path = output / "quantization_comparison.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)

    # 2. KV cache scaling
    kv = [r for r in measured if r.experiment_type == "kv_cache"]
    if kv:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        by_strat: dict = {}
        for r in kv:
            by_strat.setdefault(r.method, []).append(r)
        for strat, items in by_strat.items():
            items = sorted(items, key=lambda x: x.config.get("context_length") or 0)
            xs = [x.config.get("context_length") for x in items]
            mem = [
                (x.metrics.peak_gpu_allocated_mb or x.metrics.peak_rss_mb or 0) for x in items
            ]
            ttft = [x.metrics.ttft_ms.mean if x.metrics.ttft_ms else 0 for x in items]
            axes[0].plot(xs, mem, marker="o", label=strat)
            axes[1].plot(xs, ttft, marker="o", label=strat)
        axes[0].set_xlabel("context tokens")
        axes[0].set_ylabel("peak memory (MB)")
        axes[0].set_title("KV memory vs context")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("context tokens")
        axes[1].set_ylabel("TTFT mean (ms)")
        axes[1].set_title("Prefill vs context")
        axes[1].legend(fontsize=8)
        fig.suptitle(header)
        fig.tight_layout()
        path = output / "kv_cache_scaling.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)

    # 3. Speculative decoding
    spec = [r for r in measured if r.experiment_type == "speculative_decoding"]
    if spec:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [r.method for r in spec]
        tps = [
            r.metrics.tokens_per_sec.mean if r.metrics.tokens_per_sec else 0 for r in spec
        ]
        acc = [
            (r.metrics.extra or {}).get("acceptance_rate_mean") or 0 for r in spec
        ]
        axes[0].bar(range(len(labels)), tps, color="#2c8a5f")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=35, ha="right")
        axes[0].set_ylabel("tokens / sec")
        axes[0].set_title("Speculative throughput")
        axes[1].bar(range(len(labels)), acc, color="#5f2c8a")
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=35, ha="right")
        axes[1].set_ylabel("acceptance rate")
        axes[1].set_title("Draft acceptance")
        fig.suptitle(header)
        fig.tight_layout()
        path = output / "speculative_decoding.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)

    # 4. Pareto scatter
    front = pareto_front(measured)
    if measured:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        xs, ys, names = [], [], []
        for r in measured:
            mem = r.metrics.peak_gpu_allocated_mb or r.metrics.peak_rss_mb
            tps = r.metrics.tokens_per_sec.mean if r.metrics.tokens_per_sec else None
            if mem is None or tps is None:
                continue
            xs.append(mem)
            ys.append(tps)
            names.append(r.method)
        ax.scatter(xs, ys, c="#4a4a4a", s=36, label="measured")
        if front:
            fx, fy = [], []
            for item in front:
                obj = item["objectives"]
                if obj.get("memory_mb") is not None and obj.get("tokens_per_sec") is not None:
                    fx.append(obj["memory_mb"])
                    fy.append(obj["tokens_per_sec"])
            if fx:
                pairs = sorted(zip(fx, fy))
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-", color="#b33", label="Pareto")
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("peak memory (MB)")
        ax.set_ylabel("tokens / sec")
        ax.set_title("Throughput–memory Pareto (measured only)")
        ax.legend()
        fig.suptitle(header)
        fig.tight_layout()
        path = output / "pareto_throughput_memory.png"
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)

    # 5. Unsupported catalog (table figure)
    skipped = [r for r in recs if r.status != Status.MEASURED]
    if skipped:
        fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * len(skipped) + 1.5)))
        ax.axis("off")
        cells = [[r.method, r.status.value, (r.reason or "")[:90]] for r in skipped]
        table = ax.table(
            cellText=cells,
            colLabels=["method", "status", "reason"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        ax.set_title("Unsupported / error experiments (not scored)")
        fig.tight_layout()
        path = output / "unsupported_experiments.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def plot_search_study(study: dict, output_dir) -> List[Path]:
    """Bar charts for optimizer baselines. Uses only measured hypervolumes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = study.get("comparison") or []
    if not rows:
        return []
    names = [r["strategy"] for r in rows]
    evals = [r.get("n_evaluated") or 0 for r in rows]
    hvs = [r.get("hv_vs_grid") if r.get("hv_vs_grid") is not None else 0 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(names, evals, color="#2c5f8a")
    axes[0].set_ylabel("wall-clock evaluations")
    axes[0].set_title("Measurement budget")
    axes[1].bar(names, hvs, color="#8a3c2c")
    axes[1].set_ylabel("hypervolume / grid")
    axes[1].set_title("Throughput–memory HV vs exhaustive grid")
    fig.suptitle("InferLite search vs baselines (measured only)")
    fig.tight_layout()
    path = output / "search_vs_baselines.png"
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path]


def plot_ablation(report: dict, output_dir) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = report.get("variants") or {}
    if not variants:
        return []
    names, r2s = [], []
    for name, payload in variants.items():
        tps = ((payload.get("targets") or {}).get("tokens_per_sec") or {})
        names.append(name)
        r2s.append(tps.get("r2") if tps.get("r2") is not None else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, r2s, color="#2c8a5f")
    ax.set_ylabel("leave-one-out R² (tokens/s)")
    ax.set_title("Predictor ablation (measured rows only)")
    ax.axhline(0, color="#888", linewidth=0.8)
    fig.tight_layout()
    path = output / "predictor_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]
