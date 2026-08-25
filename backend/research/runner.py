"""Run an experiment suite from a YAML/JSON config. Never fills gaps with fake numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from research.capabilities import probe
from research.engine import run_benchmark
from research.env import collect_environment, set_seed
from research.experiments.batching import run_batching_suite
from research.experiments.kv_cache import run_kv_cache_suite
from research.experiments.pareto import pareto_front
from research.experiments.quantization import run_quantization_suite
from research.experiments.speculative import run_speculative_suite
from research.optimizer import run_search_study, serialize_study
from research.predictor import ablation_study
from research.schema import ExperimentRecord
from research.storage import ResultStore
from research.viz import plot_ablation, plot_search_study, plot_suite


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a mapping")
    return data


def run_config(
    config: Dict[str, Any],
    *,
    results_dir: Optional[Union[str, Path]] = None,
    make_plots: bool = True,
) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    env = collect_environment(seed=seed)
    model_id = config.get("model_id") or "gpt2"
    prompt = config.get("prompt") or "The future of efficient language model inference is"
    max_new = int(config.get("max_new_tokens", 32))
    warmup = int(config.get("warmup_runs", 1))
    runs = int(config.get("measure_runs", 3))
    store = ResultStore(results_dir or config.get("results_dir"))
    records: List[ExperimentRecord] = []
    plots: List[str] = []

    experiments = config.get("experiments") or [{"type": "quantization"}]
    for spec in experiments:
        kind = (spec.get("type") or spec.get("name") or "").lower()
        if kind in {"baseline", "benchmark"}:
            records.append(
                run_benchmark(
                    model_id=spec.get("model_id", model_id),
                    method=spec.get("method", spec.get("precision", "fp32")),
                    backend=spec.get("backend", "transformers"),
                    prompt=spec.get("prompt", prompt),
                    max_new_tokens=int(spec.get("max_new_tokens", max_new)),
                    warmup_runs=int(spec.get("warmup_runs", warmup)),
                    measure_runs=int(spec.get("measure_runs", runs)),
                    seed=seed,
                    environment=env,
                    **{
                        k: v
                        for k, v in {**config, **spec}.items()
                        if k in {"gguf_file", "filename", "model_path", "quantized_model_id", "n_ctx", "n_gpu_layers"}
                    },
                )
            )
        elif kind in {"quantization", "quantize"}:
            records.extend(
                run_quantization_suite(
                    spec.get("model_id", model_id),
                    methods=spec.get("methods"),
                    prompt=spec.get("prompt", prompt),
                    max_new_tokens=int(spec.get("max_new_tokens", max_new)),
                    warmup_runs=int(spec.get("warmup_runs", warmup)),
                    measure_runs=int(spec.get("measure_runs", runs)),
                    seed=seed,
                    gguf_repo=spec.get("gguf_repo") or config.get("gguf_repo"),
                    gguf_file=spec.get("gguf_file") or config.get("gguf_file"),
                    gptq_model_id=spec.get("gptq_model_id") or config.get("gptq_model_id"),
                    awq_model_id=spec.get("awq_model_id") or config.get("awq_model_id"),
                )
            )
        elif kind in {"kv_cache", "kv-cache", "kv"}:
            records.extend(
                run_kv_cache_suite(
                    spec.get("model_id", model_id),
                    context_lengths=spec.get("context_lengths"),
                    max_new_tokens=int(spec.get("max_new_tokens", min(16, max_new))),
                    measure_runs=int(spec.get("measure_runs", max(1, runs))),
                    seed=seed,
                    strategies=spec.get("strategies"),
                )
            )
        elif kind in {"speculative", "speculative_decoding"}:
            records.extend(
                run_speculative_suite(
                    spec.get("target_model_id") or spec.get("model_id", model_id),
                    spec.get("draft_model_id") or config.get("draft_model_id") or "distilgpt2",
                    gammas=spec.get("gammas"),
                    max_new_tokens=int(spec.get("max_new_tokens", max_new)),
                    measure_runs=int(spec.get("measure_runs", max(1, runs))),
                    seed=seed,
                    prompt=spec.get("prompt", prompt),
                )
            )
        elif kind in {"batching", "continuous_batching"}:
            records.extend(
                run_batching_suite(
                    spec.get("model_id", model_id),
                    num_requests=int(spec.get("num_requests", 4)),
                    max_batch_size=int(spec.get("max_batch_size", 2)),
                    max_new_tokens=int(spec.get("max_new_tokens", min(8, max_new))),
                    seed=seed,
                )
            )
        elif kind in {"search", "optimize", "optimizer"}:
            study = run_search_study(
                {
                    **config,
                    **{k: spec.get(k) for k in ("budget", "strategies", "search_space") if spec.get(k) is not None},
                }
            )
            grid_payload = (study.get("strategies") or {}).get("grid") or {}
            records.extend(grid_payload.get("records") or [])
            store.root.mkdir(parents=True, exist_ok=True)
            import json as _json

            (store.root / "search_study.json").write_text(
                _json.dumps(serialize_study(study), indent=2, default=str),
                encoding="utf-8",
            )
            if make_plots:
                plots.extend(str(p) for p in plot_search_study(study, store.root / "figures"))
                measured = [r for r in records if r.status.value == "measured"]
                if len(measured) >= 3:
                    plots.extend(str(p) for p in plot_ablation(ablation_study(measured), store.root / "figures"))
        else:
            raise ValueError(f"unknown experiment type: {kind}")

    store.save_many(records)
    bundle = store.write_bundle(records, name=config.get("name") or "suite")
    if make_plots:
        plots.extend(
            str(p)
            for p in plot_suite(records, store.root / "figures", title_suffix=config.get("name") or "")
        )
    front = pareto_front(records)
    return {
        "config_name": config.get("name"),
        "n_records": len(records),
        "n_measured": sum(1 for r in records if r.status.value == "measured"),
        "n_unsupported": sum(1 for r in records if r.status.value == "unsupported"),
        "n_error": sum(1 for r in records if r.status.value == "error"),
        "bundle": str(bundle),
        "csv": str(store.csv_path),
        "plots": plots,
        "pareto": [{k: v for k, v in item.items() if k != "record"} for item in front],
        "records": [r.model_dump() for r in records],
        "capabilities": probe(),
    }
