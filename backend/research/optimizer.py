"""Multi-objective search: grid, random, hardware heuristic, InferLite (surrogate).

Every scored point is a wall-clock `run_benchmark` (or an explicit unsupported).
The queueing simulator is not used here.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from research.capabilities import probe
from research.energy import probe_energy
from research.engine import run_benchmark
from research.env import collect_environment, set_seed
from research.experiments.pareto import hypervolume_throughput_memory, pareto_front
from research.predictor import PerformancePredictor
from research.schema import ExperimentRecord, Status
from research.search_space import Candidate, from_config
from research.workloads import prompt_for_tokens

EvalFn = Callable[[Candidate], ExperimentRecord]


def _objectives(rec: ExperimentRecord) -> Optional[Tuple[float, float, float]]:
    if rec.status != Status.MEASURED or rec.metrics is None:
        return None
    p95 = rec.metrics.e2e_latency_ms.p95 if rec.metrics.e2e_latency_ms else None
    tps = rec.metrics.tokens_per_sec.mean if rec.metrics.tokens_per_sec else None
    mem = rec.metrics.peak_gpu_allocated_mb
    if mem is None:
        mem = rec.metrics.peak_rss_mb
    if p95 is None or tps is None or mem is None:
        return None
    return float(p95), float(mem), float(tps)


def hardware_heuristic(candidates: Sequence[Candidate], env: Optional[Dict[str, Any]] = None) -> Candidate:
    """One config from hardware facts. Not a scored benchmark by itself."""
    env = env or collect_environment()
    torch_info = env.get("torch") or {}
    cuda = bool(torch_info.get("cuda_available"))
    mps = bool(torch_info.get("mps_available"))
    gpu = torch_info.get("gpu") or {}
    gpu_mem = float(gpu.get("total_memory_mb") or 0)
    methods = {c.method for c in candidates}

    if cuda and gpu_mem and gpu_mem < 10_000 and "int4_bnb" in methods:
        prefer = "int4_bnb"
    elif (cuda or mps) and "fp16" in methods:
        prefer = "fp16"
    elif "fp32" in methods:
        prefer = "fp32"
    else:
        prefer = next(iter(methods))

    pool = [c for c in candidates if c.method == prefer] or list(candidates)
    pool.sort(key=lambda c: (c.context_tokens, c.max_new_tokens, c.batch_size))
    # Prefer a mid-context point when several exist
    return pool[len(pool) // 2]


def _diverse_seed(candidates: Sequence[Candidate], k: int, rng: random.Random) -> List[Candidate]:
    if k >= len(candidates):
        return list(candidates)
    chosen = [rng.choice(list(candidates))]
    rest = [c for c in candidates if c.key != chosen[0].key]
    while len(chosen) < k and rest:
        def dist(c: Candidate) -> int:
            return min(
                abs(c.context_tokens - x.context_tokens)
                + abs(c.max_new_tokens - x.max_new_tokens)
                + 50 * (c.method != x.method)
                + 10 * abs(c.batch_size - x.batch_size)
                for x in chosen
            )
        rest.sort(key=dist, reverse=True)
        nxt = rest.pop(0)
        chosen.append(nxt)
    return chosen


def _predicted_record(cand: Candidate, pred_vec: Sequence[float], template: ExperimentRecord) -> ExperimentRecord:
    """Build a synthetic *prediction* record for acquisition only. Never persisted as measured."""
    from research.schema import BenchmarkMetrics, PercentileStats

    def _stat(val: float) -> PercentileStats:
        return PercentileStats(
            n=0, mean=float(val), std=0, min=float(val), p50=float(val),
            p95=float(val), p99=float(val), max=float(val),
        )

    p95, tps, mem = pred_vec
    return ExperimentRecord(
        experiment_id=f"pred_{cand.key}",
        experiment_type="search_prediction",
        status=Status.MEASURED,
        timestamp_utc=template.timestamp_utc,
        model_id=template.model_id,
        backend=template.backend,
        method=cand.method,
        device=template.device,
        precision=cand.method,
        config=cand.as_dict(),
        environment=template.environment,
        metrics=BenchmarkMetrics(
            e2e_latency_ms=_stat(p95),
            tokens_per_sec=_stat(tps),
            peak_rss_mb=float(mem),
        ),
        notes=["predictor output — not a wall-clock measurement"],
    )


def inferlite_order(
    remaining: Sequence[Candidate],
    measured: Sequence[ExperimentRecord],
    rng: random.Random,
) -> List[Candidate]:
    """Score unevaluated configs with a ridge surrogate; prefer predicted Pareto + diversity."""
    if not remaining:
        return []
    if len(measured) < 3:
        return _diverse_seed(remaining, len(remaining), rng)
    model = PerformancePredictor().fit(measured)
    # dummy records for featurization
    probes = []
    for cand in remaining:
        rec = ExperimentRecord(
            experiment_id=cand.key,
            experiment_type="search",
            status=Status.MEASURED,
            timestamp_utc=measured[0].timestamp_utc,
            model_id=measured[0].model_id,
            backend=measured[0].backend,
            method=cand.method,
            device=measured[0].device,
            precision=cand.method,
            config={
                "context_tokens": cand.context_tokens,
                "max_new_tokens": cand.max_new_tokens,
                "batch_size": cand.batch_size,
            },
            environment=measured[0].environment,
            metrics=measured[0].metrics,
        )
        probes.append(rec)
    preds = model.predict_records(probes)
    scored: List[Tuple[float, Candidate]] = []
    meas_front = [
        r
        for r in measured
        if _objectives(r) is not None
    ]
    for cand, rec, vec in zip(remaining, probes, preds):
        if not np_all_finite(vec):
            scored.append((0.0, cand))
            continue
        pred_rec = _predicted_record(cand, vec, measured[0])
        combined = list(meas_front) + [pred_rec]
        front = pareto_front(combined)
        on_front = any(item["experiment_id"] == pred_rec.experiment_id for item in front)
        # distance in (mem, tps) to nearest measured point
        p95, tps, mem = float(vec[0]), float(vec[1]), float(vec[2])
        dist = min(
            abs(mem - (_objectives(m)[1])) + abs(tps - (_objectives(m)[2]))
            for m in meas_front
        ) if meas_front else 1.0
        score = (10.0 if on_front else 0.0) + dist
        scored.append((score, cand))
        _ = p95
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]


def np_all_finite(vec: Sequence[float]) -> bool:
    import math

    return all(v is not None and math.isfinite(float(v)) for v in vec)


def make_eval_fn(
    *,
    model_id: str,
    seed: int = 42,
    warmup_runs: int = 1,
    measure_runs: int = 3,
    tokenizer: Any = None,
    loaded_by_method: Optional[Dict[str, Any]] = None,
    extra_load: Optional[Dict[str, Any]] = None,
) -> EvalFn:
    """Wall-clock evaluator. Reuses a loaded model per method when provided."""
    extra_load = extra_load or {}
    cache = loaded_by_method if loaded_by_method is not None else {}

    def _eval(cand: Candidate) -> ExperimentRecord:
        prompt = prompt_for_tokens(cand.context_tokens, tokenizer=tokenizer)
        loaded = cache.get(cand.method)
        rec = run_benchmark(
            model_id=model_id,
            method=cand.method,
            prompt=prompt,
            max_new_tokens=cand.max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            seed=seed,
            experiment_type="search",
            loaded=loaded,
            model=extra_load.get("model") if cand.method in {"fp32", "fp16"} else extra_load.get("model"),
            tokenizer=extra_load.get("tokenizer"),
            config={
                "context_tokens": cand.context_tokens,
                "max_new_tokens": cand.max_new_tokens,
                "batch_size": cand.batch_size,
                "workload": cand.workload,
                "candidate_key": cand.key,
            },
        )
        rec.config = {**(rec.config or {}), **cand.as_dict()}
        rec.notes = list(rec.notes or [])
        rec.notes.append("search: wall-clock measurement" if rec.status == Status.MEASURED else f"search: {rec.status.value}")
        energy = probe_energy()
        if rec.metrics is not None:
            rec.metrics.extra = {**(rec.metrics.extra or {}), "energy_probe": energy}
            if not energy.get("supported"):
                rec.notes.append(energy.get("reason") or "energy unsupported")
        if rec.status == Status.MEASURED and loaded is None and extra_load.get("cache_loads"):
            pass
        return rec

    return _eval


def run_strategy(
    name: str,
    candidates: Sequence[Candidate],
    evaluate: EvalFn,
    *,
    budget: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    pool = list(candidates)
    if not pool:
        return {
            "strategy": name,
            "n_candidates": 0,
            "n_measured": 0,
            "records": [],
            "reason": "empty search space after capability filter",
        }

    chosen: List[Candidate] = []
    if name == "grid":
        chosen = pool
    elif name == "random":
        k = min(budget or len(pool), len(pool))
        chosen = rng.sample(pool, k)
    elif name == "heuristic":
        chosen = [hardware_heuristic(pool)]
    elif name == "inferlite":
        k = min(budget or len(pool), len(pool))
        seed_n = min(3, k)
        chosen = _diverse_seed(pool, seed_n, rng)
        # heuristic always included if present
        h = hardware_heuristic(pool)
        if h.key not in {c.key for c in chosen}:
            chosen[0] = h
    else:
        raise ValueError(f"unknown strategy: {name}")

    records: List[ExperimentRecord] = []
    seen = set()
    for cand in chosen:
        rec = evaluate(cand)
        records.append(rec)
        seen.add(cand.key)

    if name == "inferlite":
        k = min(budget or len(pool), len(pool))
        remaining = [c for c in pool if c.key not in seen]
        while len(records) < k and remaining:
            measured_now = [r for r in records if r.status == Status.MEASURED]
            order = inferlite_order(remaining, measured_now, rng)
            nxt = order[0] if order else remaining[0]
            records.append(evaluate(nxt))
            seen.add(nxt.key)
            remaining = [c for c in remaining if c.key not in seen]

    measured = [r for r in records if r.status == Status.MEASURED]
    return {
        "strategy": name,
        "n_candidates": len(pool),
        "n_evaluated": len(records),
        "n_measured": len(measured),
        "n_unsupported": sum(1 for r in records if r.status == Status.UNSUPPORTED),
        "budget": budget,
        "hypervolume": hypervolume_throughput_memory(measured),
        "pareto": [
            {k: v for k, v in item.items() if k != "record"}
            for item in pareto_front(measured)
        ],
        "records": records,
        "simulation": False,
    }


def compare_strategies(
    candidates: Sequence[Candidate],
    evaluate: EvalFn,
    *,
    budget: Optional[int] = None,
    seed: int = 42,
    strategies: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    names = list(strategies or ("grid", "random", "heuristic", "inferlite"))
    grid_budget = len(candidates)
    inf_budget = budget if budget is not None else max(1, min(grid_budget, max(3, grid_budget // 2)))
    results = {}
    for name in names:
        b = None if name == "grid" else (1 if name == "heuristic" else inf_budget)
        results[name] = run_strategy(name, candidates, evaluate, budget=b, seed=seed)

    grid_recs = results.get("grid", {}).get("records") or []
    grid_hv = results.get("grid", {}).get("hypervolume")
    comparison = []
    for name, payload in results.items():
        hv = payload.get("hypervolume")
        comparison.append(
            {
                "strategy": name,
                "n_evaluated": payload.get("n_evaluated"),
                "n_measured": payload.get("n_measured"),
                "hypervolume": hv,
                "hv_vs_grid": (
                    None
                    if grid_hv in (None, 0) or hv is None
                    else float(hv) / float(grid_hv)
                ),
                "simulation": False,
            }
        )
    return {
        "research_question": (
            "Can we find a strong LLM inference configuration for given hardware "
            "and workload without exhaustive benchmarking?"
        ),
        "n_search_space": len(candidates),
        "inferlite_budget": inf_budget,
        "energy": probe_energy(),
        "device": probe().get("device"),
        "comparison": comparison,
        "strategies": results,
        "grid_n": len(grid_recs),
        "simulation": False,
    }


def run_search_study(config: Dict[str, Any], *, evaluate: Optional[EvalFn] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    candidates = from_config(config)
    budget = config.get("budget")
    strategies = config.get("strategies")
    if evaluate is None:
        evaluate = make_eval_fn(
            model_id=config.get("model_id") or "gpt2",
            seed=seed,
            warmup_runs=int(config.get("warmup_runs", 1)),
            measure_runs=int(config.get("measure_runs", 2)),
            extra_load={
                "model": config.get("_model"),
                "tokenizer": config.get("_tokenizer"),
            },
        )
    return compare_strategies(
        candidates,
        evaluate,
        budget=int(budget) if budget is not None else None,
        seed=seed,
        strategies=strategies,
    )


def serialize_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe study dump. Records are summarized, never filled in."""
    out = {k: v for k, v in study.items() if k != "strategies"}
    strats = {}
    for name, payload in (study.get("strategies") or {}).items():
        recs = payload.get("records") or []
        strats[name] = {
            k: v
            for k, v in payload.items()
            if k != "records"
        }
        strats[name]["records"] = [r.model_dump() for r in recs]
    out["strategies"] = strats
    return out
