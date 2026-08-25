"""InferLite research CLI. Local measurements; does not require the API server."""

from __future__ import annotations

import json
from pathlib import Path

import click

from research.capabilities import probe
from research.engine import run_benchmark
from research.env import collect_environment
from research.experiments.batching import run_batching_suite
from research.experiments.kv_cache import run_kv_cache_suite
from research.experiments.pareto import pareto_front
from research.experiments.quantization import run_quantization_suite
from research.experiments.speculative import run_speculative_suite
from research.predictor import ablation_study
from research.runner import load_config, run_config
from research.storage import ResultStore
from research.viz import plot_suite


def _dump(obj) -> None:
    click.echo(json.dumps(obj, indent=2, default=str))


def _store(results_dir: str | None) -> ResultStore:
    return ResultStore(results_dir)


@click.group()
def cli():
    """InferLite: reproducible LLM inference research (measured or explicitly unsupported)."""


@cli.command("env")
def env_cmd():
    """Print hardware/software metadata used for reproducibility."""
    _dump(collect_environment())


@cli.command()
def capabilities():
    """Show which experiments this machine can actually run."""
    _dump(probe())


@cli.command()
@click.option("--model", default="gpt2", show_default=True)
@click.option("--method", default="fp32", show_default=True)
@click.option("--backend", default="transformers", show_default=True)
@click.option("--max-new", default=32, show_default=True)
@click.option("--runs", default=3, show_default=True)
@click.option("--warmup", default=1, show_default=True)
@click.option("--results-dir", default=None)
@click.option("--gguf-file", default=None)
def bench(model, method, backend, max_new, runs, warmup, results_dir, gguf_file):
    """Run a single wall-clock generation benchmark."""
    rec = run_benchmark(
        model_id=model,
        method=method,
        backend=backend,
        max_new_tokens=max_new,
        measure_runs=runs,
        warmup_runs=warmup,
        gguf_file=gguf_file,
        filename=gguf_file,
    )
    _store(results_dir).save(rec)
    _dump(rec.model_dump())


@cli.command("quant")
@click.option("--model", default="gpt2", show_default=True)
@click.option("--methods", default=None, help="Comma-separated methods")
@click.option("--max-new", default=32, show_default=True)
@click.option("--runs", default=3, show_default=True)
@click.option("--gguf-repo", default=None)
@click.option("--gguf-file", default=None)
@click.option("--gptq-model", default=None)
@click.option("--awq-model", default=None)
@click.option("--results-dir", default=None)
def quant(model, methods, max_new, runs, gguf_repo, gguf_file, gptq_model, awq_model, results_dir):
    """Run INT8/INT4/AWQ/GPTQ/GGUF experiments. Unsupported methods are labeled, not scored."""
    method_list = [m.strip() for m in methods.split(",")] if methods else None
    recs = run_quantization_suite(
        model,
        methods=method_list,
        max_new_tokens=max_new,
        measure_runs=runs,
        gguf_repo=gguf_repo,
        gguf_file=gguf_file,
        gptq_model_id=gptq_model,
        awq_model_id=awq_model,
    )
    store = _store(results_dir)
    store.save_many(recs)
    store.write_bundle(recs, name="quantization")
    _dump([r.model_dump() for r in recs])


@cli.command("kv-cache")
@click.option("--model", default="gpt2", show_default=True)
@click.option("--lengths", default="32,64,128", show_default=True)
@click.option("--max-new", default=16, show_default=True)
@click.option("--results-dir", default=None)
def kv_cache(model, lengths, max_new, results_dir):
    """Measure KV-cache memory and TTFT vs context length."""
    ctx = [int(x) for x in lengths.split(",") if x.strip()]
    recs = run_kv_cache_suite(model, context_lengths=ctx, max_new_tokens=max_new)
    _store(results_dir).save_many(recs)
    _dump([r.model_dump() for r in recs])


@cli.command()
@click.option("--target", default="gpt2", show_default=True)
@click.option("--draft", default="distilgpt2", show_default=True)
@click.option("--gammas", default="2,4", show_default=True)
@click.option("--max-new", default=32, show_default=True)
@click.option("--results-dir", default=None)
def speculative(target, draft, gammas, max_new, results_dir):
    """Greedy speculative decoding vs autoregressive baseline."""
    g = [int(x) for x in gammas.split(",") if x.strip()]
    recs = run_speculative_suite(target, draft, gammas=g, max_new_tokens=max_new)
    _store(results_dir).save_many(recs)
    _dump([r.model_dump() for r in recs])


@cli.command()
@click.option("--model", default="gpt2", show_default=True)
@click.option("--requests", default=4, show_default=True)
@click.option("--batch-size", default=2, show_default=True)
@click.option("--max-new", default=8, show_default=True)
@click.option("--results-dir", default=None)
def batching(model, requests, batch_size, max_new, results_dir):
    """Compare static batching with in-process continuous batching."""
    recs = run_batching_suite(
        model, num_requests=requests, max_batch_size=batch_size, max_new_tokens=max_new
    )
    _store(results_dir).save_many(recs)
    _dump([r.model_dump() for r in recs])


@cli.command()
@click.option("--results-dir", default=None)
@click.option("--bundle", default=None, help="Path to a suite JSON bundle")
def pareto(results_dir, bundle):
    """Pareto front over measured records (latency, memory, throughput)."""
    store = _store(results_dir)
    recs = store.load_json_file(bundle) if bundle else store.load_all()
    front = pareto_front(recs)
    _dump([{k: v for k, v in item.items() if k != "record"} for item in front])


@cli.command("plot")
@click.option("--results-dir", default=None)
@click.option("--bundle", default=None)
@click.option("--out", default=None)
def plot(results_dir, bundle, out):
    """Write research figures from stored measured results."""
    store = _store(results_dir)
    recs = store.load_json_file(bundle) if bundle else store.load_all()
    paths = plot_suite(recs, out or (store.root / "figures"))
    _dump({"figures": [str(p) for p in paths]})


@cli.command("suite")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--results-dir", default=None)
@click.option("--no-plots", is_flag=True)
def suite(config_path, results_dir, no_plots):
    """Run a YAML/JSON experiment config end-to-end."""
    cfg = load_config(config_path)
    summary = run_config(cfg, results_dir=results_dir, make_plots=not no_plots)
    _dump({k: v for k, v in summary.items() if k != "records"})


@cli.command("optimize")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--results-dir", default=None)
@click.option("--no-plots", is_flag=True)
def optimize(config_path, results_dir, no_plots):
    """Grid vs random vs heuristic vs InferLite. Wall-clock measurements only.

    YAML `seeds` + `budgets` measures the grid once, then replays budgeted
    strategies (mean / std / 95% CI of hypervolume vs random).
    """
    cfg = load_config(config_path)
    if not any((e.get("type") or "") in {"search", "optimize", "optimizer"} for e in (cfg.get("experiments") or [])):
        cfg.setdefault("experiments", []).append({"type": "search"})
    if results_dir:
        cfg["results_dir"] = results_dir
    summary = run_config(cfg, results_dir=results_dir or cfg.get("results_dir"), make_plots=not no_plots)
    _dump({k: v for k, v in summary.items() if k != "records"})


@cli.command("predict")
@click.option("--bundle", required=True, type=click.Path(exists=True), help="JSON bundle of measured records")
@click.option("--out", default=None)
def predict(bundle, out):
    """Leave-one-out predictor + feature-group ablation on measured rows."""
    store = ResultStore(None)
    recs = store.load_json_file(bundle)
    measured = [r for r in recs if r.status.value == "measured"]
    report = ablation_study(measured)
    if out:
        from pathlib import Path
        from research.viz import plot_ablation

        paths = plot_ablation(report, out)
        report["plots"] = [str(p) for p in paths]
    _dump(report)


@cli.command("simulate")
@click.option("--rps", default=10.0)
@click.option("--duration", default=5)
@click.option("--nodes", default=1)
def simulate(rps, duration, nodes):
    """Queueing simulation for capacity planning. This is NOT a model benchmark."""
    import asyncio

    from services.simulator import ServingSimulator, SimulationConfig

    click.echo(
        "NOTE: this is a discrete-event serving simulation, not a measured LLM run.",
        err=True,
    )
    cfg = SimulationConfig(
        rps=rps, duration_seconds=duration, nodes=nodes, tokens_per_request=128
    )
    result = asyncio.run(ServingSimulator(cfg).run_simulation())
    _dump(result.model_dump())


def main():
    cli()


if __name__ == "__main__":
    main()
