"""Multi-objective search: grid, random, hardware heuristic, InferLite (surrogate).

Every scored point is a wall-clock `run_benchmark` (or an explicit unsupported).
The queueing simulator is not used here.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from research.backends import try_load
from research.capabilities import probe
from research.energy import probe_energy
from research.engine import run_benchmark
from research.env import collect_environment, select_device, set_seed
from research.experiments.pareto import hypervolume_throughput_memory, pareto_front
from research.metrics import mean_std_ci95
from research.predictor import PerformancePredictor
from research.schema import ExperimentRecord, Status
from research.search_space import Candidate, from_config
from research.workloads import prompt_for_tokens

EvalFn = Callable[[Candidate], ExperimentRecord]

RESEARCH_QUESTION = (
    "Can a hardware-aware multi-objective optimizer identify near-Pareto-optimal "
    "LLM inference configurations using substantially fewer measurements than "
    "exhaustive search?"
)
DEFAULT_SEEDS = (42, 123, 456, 789, 1000)
DEFAULT_BUDGETS = (2, 4, 8, 16)


def _free_memory() -> None:
    """Drop CPU/GPU caches after a wall-clock job. Does not invent metrics."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def cached_evaluate(evaluate: EvalFn) -> EvalFn:
    """Reuse a measured candidate across grid/random/heuristic/InferLite. One load per key."""
    store: Dict[str, ExperimentRecord] = {}

    def _inner(cand: Candidate) -> ExperimentRecord:
        hit = store.get(cand.key)
        if hit is not None:
            return hit
        rec = evaluate(cand)
        store[cand.key] = rec
        _free_memory()
        return rec

    return _inner


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


def _evict_other_methods(cache: Dict[str, Any], keep: str) -> None:
    for method, loaded in list(cache.items()):
        if method == keep:
            continue
        closer = getattr(loaded, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        cache.pop(method, None)
    _free_memory()


def close_loaded(cache: Optional[Dict[str, Any]]) -> None:
    if not cache:
        return
    for loaded in list(cache.values()):
        closer = getattr(loaded, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
    cache.clear()
    _free_memory()


def make_eval_fn(
    *,
    model_id: str,
    seed: int = 42,
    warmup_runs: int = 1,
    measure_runs: int = 3,
    tokenizer: Any = None,
    loaded_by_method: Optional[Dict[str, Any]] = None,
    extra_load: Optional[Dict[str, Any]] = None,
    keep_one_method: bool = False,
) -> EvalFn:
    """Wall-clock evaluator. Loads each precision once and reuses it across candidates."""
    extra_load = extra_load or {}
    cache = loaded_by_method if loaded_by_method is not None else {}

    def _eval(cand: Candidate) -> ExperimentRecord:
        prompt = prompt_for_tokens(cand.context_tokens, tokenizer=tokenizer or extra_load.get("tokenizer"))
        method = cand.method
        backend = "transformers"
        mid = model_id
        extra_kw: Dict[str, Any] = {}
        if method.lower() in {"gguf", "gguf_q4_k_m", "llama.cpp"}:
            backend = "llama.cpp"
            mid = extra_load.get("gguf_repo") or model_id
            extra_kw["gguf_file"] = extra_load.get("gguf_file")
            extra_kw["filename"] = extra_load.get("gguf_file") or extra_load.get("filename")
            extra_kw["n_ctx"] = int(
                extra_load.get("n_ctx") or max(256, cand.context_tokens + cand.max_new_tokens + 32)
            )
        injected = extra_load.get("model")
        loaded = cache.get(cand.method)
        if loaded is None and injected is None:
            if keep_one_method:
                _evict_other_methods(cache, cand.method)
            loaded, _err = try_load(method, mid, device=select_device(), **extra_kw)
            if loaded is not None:
                cache[cand.method] = loaded
        rec = run_benchmark(
            model_id=mid,
            method=method,
            backend=backend,
            prompt=prompt,
            max_new_tokens=cand.max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            seed=seed,
            experiment_type="search",
            loaded=loaded,
            model=injected,
            tokenizer=extra_load.get("tokenizer"),
            config={
                "context_tokens": cand.context_tokens,
                "max_new_tokens": cand.max_new_tokens,
                "batch_size": cand.batch_size,
                "workload": cand.workload,
                "candidate_key": cand.key,
            },
            **extra_kw,
        )
        rec.config = {**(rec.config or {}), **cand.as_dict()}
        rec.notes = list(rec.notes or [])
        rec.notes.append("search: wall-clock measurement" if rec.status == Status.MEASURED else f"search: {rec.status.value}")
        energy = probe_energy()
        if rec.metrics is not None:
            rec.metrics.extra = {**(rec.metrics.extra or {}), "energy_probe": energy}
            if not energy.get("supported"):
                rec.notes.append(energy.get("reason") or "energy unsupported")
        _free_memory()
        return rec

    return _eval


def run_strategy(
    name: str,
    candidates: Sequence[Candidate],
    evaluate: EvalFn,
    *,
    budget: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
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
        if verbose:
            print(f"[search {name}] {cand.key}", flush=True)
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
            if verbose:
                print(f"[search {name}] {nxt.key}", flush=True)
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
    evaluate = cached_evaluate(evaluate)
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
        "research_question": RESEARCH_QUESTION,
        "n_search_space": len(candidates),
        "inferlite_budget": inf_budget,
        "energy": probe_energy(),
        "device": probe().get("device"),
        "comparison": comparison,
        "strategies": results,
        "grid_n": len(grid_recs),
        "simulation": False,
    }


def _strategy_budget(name: str, budget: Optional[int], n_space: int) -> Optional[int]:
    if name == "grid":
        return None
    if name == "heuristic":
        return 1
    return budget if budget is not None else max(1, min(n_space, max(3, n_space // 2)))


def _hv_ratio(hv: Optional[float], grid_hv: Optional[float]) -> Optional[float]:
    if grid_hv in (None, 0) or hv is None:
        return None
    return float(hv) / float(grid_hv)


def run_budget_sweep(
    candidates: Sequence[Candidate],
    evaluate: EvalFn,
    *,
    seeds: Sequence[int],
    budgets: Sequence[int],
    strategies: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Measure the exhaustive grid once, then replay budgeted strategies from cache.

    Random / InferLite / heuristic never invent metrics: each pick is a previously
    timed wall-clock record. InferLite's sequential acquisition still uses the
    measured values of points it has already chosen.
    """
    names = list(strategies or ("grid", "random", "heuristic", "inferlite"))
    evaluate = cached_evaluate(evaluate)
    seed_list = [int(s) for s in seeds]
    n_space = len(candidates)
    budget_list = sorted({max(1, min(int(b), n_space)) for b in budgets})

    print(f"[search grid] measuring {n_space} configurations once (wall-clock)", flush=True)
    grid = run_strategy("grid", candidates, evaluate, budget=None, seed=seed_list[0], verbose=True)
    grid_hv = grid.get("hypervolume")
    grid_n = grid.get("n_measured") or 0

    budgeted = [n for n in names if n != "grid"]
    sweep = []
    for budget in budget_list:
        by_strat: Dict[str, Any] = {}
        for name in budgeted:
            hvs: List[Optional[float]] = []
            ratios: List[Optional[float]] = []
            n_evals: List[int] = []
            for seed in seed_list:
                b = _strategy_budget(name, budget, n_space)
                payload = run_strategy(
                    name, candidates, evaluate, budget=b, seed=seed, verbose=False
                )
                hv = payload.get("hypervolume")
                hvs.append(hv)
                ratios.append(_hv_ratio(hv, grid_hv))
                n_evals.append(int(payload.get("n_evaluated") or 0))
            by_strat[name] = {
                "n_evaluated": n_evals[0] if n_evals else 0,
                "hypervolume": mean_std_ci95(hvs),
                "hv_vs_grid": mean_std_ci95(ratios),
                "per_seed": [
                    {
                        "seed": seed,
                        "hypervolume": hv,
                        "hv_vs_grid": ratio,
                        "n_evaluated": n_eval,
                    }
                    for seed, hv, ratio, n_eval in zip(seed_list, hvs, ratios, n_evals)
                ],
            }
        sweep.append(
            {
                "budget": budget,
                "n_search_space": n_space,
                "grid_hypervolume": grid_hv,
                "strategies": by_strat,
            }
        )

    highlight = 4 if 4 in budget_list else (budget_list[len(budget_list) // 2] if budget_list else n_space)
    highlight_row = next((row for row in sweep if row["budget"] == highlight), sweep[-1] if sweep else None)
    comparison = [
        {
            "strategy": "grid",
            "n_evaluated": grid.get("n_evaluated"),
            "n_measured": grid_n,
            "hypervolume": grid_hv,
            "hv_vs_grid": 1.0 if grid_hv is not None else None,
            "hv_mean": grid_hv,
            "hv_std": 0.0,
            "hv_ci95_low": grid_hv,
            "hv_ci95_high": grid_hv,
            "seeds": seed_list,
            "note": "exhaustive wall-clock grid; one measurement per configuration",
        }
    ]
    if highlight_row:
        for name in budgeted:
            payload = highlight_row["strategies"][name]
            hv = payload["hypervolume"]
            ratio = payload["hv_vs_grid"]
            comparison.append(
                {
                    "strategy": name,
                    "n_evaluated": payload.get("n_evaluated"),
                    "n_measured": payload.get("n_evaluated"),
                    "hypervolume": hv.get("mean"),
                    "hv_vs_grid": ratio.get("mean"),
                    "hv_mean": hv.get("mean"),
                    "hv_std": hv.get("std"),
                    "hv_ci95_low": hv.get("ci95_low"),
                    "hv_ci95_high": hv.get("ci95_high"),
                    "hv_vs_grid_mean": ratio.get("mean"),
                    "hv_vs_grid_std": ratio.get("std"),
                    "hv_vs_grid_ci95_low": ratio.get("ci95_low"),
                    "hv_vs_grid_ci95_high": ratio.get("ci95_high"),
                    "seeds": seed_list,
                    "highlight_budget": highlight,
                    "note": f"mean ± 95% t-interval over {len(seed_list)} seeds at budget {highlight}",
                }
            )

    inferlite_beats = []
    for row in sweep:
        inf = ((row["strategies"].get("inferlite") or {}).get("hv_vs_grid") or {}).get("mean")
        rnd = ((row["strategies"].get("random") or {}).get("hv_vs_grid") or {}).get("mean")
        inferlite_beats.append(
            {
                "budget": row["budget"],
                "inferlite_hv_vs_grid_mean": inf,
                "random_hv_vs_grid_mean": rnd,
                "inferlite_beats_random_on_mean": (
                    None if inf is None or rnd is None else bool(inf > rnd)
                ),
            }
        )

    return {
        "research_question": RESEARCH_QUESTION,
        "n_search_space": n_space,
        "seeds": seed_list,
        "budgets": budget_list,
        "highlight_budget": highlight,
        "energy": probe_energy(),
        "device": probe().get("device"),
        "comparison": comparison,
        "sweep": sweep,
        "inferlite_vs_random": inferlite_beats,
        "grid": grid,
        "strategies": {"grid": grid},
        "grid_n": grid.get("n_evaluated"),
        "simulation": False,
        "replay": True,
        "note": (
            "Grid is measured once on the wall clock. Random, InferLite, and the "
            "heuristic replay those records at each seed×budget; they do not invent "
            "tokens/s or latency. InferLite still chooses sequentially: the surrogate "
            "only sees points it has already picked. On this design InferLite uses "
            "the ridge model only after three measured points, so budget 2 is a "
            "diverse/heuristic seed, not surrogate search."
        ),
    }


def run_search_study(config: Dict[str, Any], *, evaluate: Optional[EvalFn] = None) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    candidates = from_config(config)
    budget = config.get("budget")
    strategies = config.get("strategies")
    seeds = config.get("seeds")
    budgets = config.get("budgets")
    loaded_by_method: Dict[str, Any] = {}
    owns_eval = evaluate is None
    if evaluate is None:
        evaluate = make_eval_fn(
            model_id=config.get("model_id") or "gpt2",
            seed=seed,
            warmup_runs=int(config.get("warmup_runs", 1)),
            measure_runs=int(config.get("measure_runs", 2)),
            loaded_by_method=loaded_by_method,
            keep_one_method=bool(config.get("keep_one_method", False)),
            extra_load={
                "model": config.get("_model"),
                "tokenizer": config.get("_tokenizer"),
                "gguf_repo": config.get("gguf_repo"),
                "gguf_file": config.get("gguf_file"),
                "filename": config.get("gguf_file"),
                "n_ctx": config.get("n_ctx"),
            },
        )
    try:
        if seeds or budgets:
            seed_list = [int(s) for s in (seeds or [seed])]
            budget_list = [int(b) for b in (budgets or ([budget] if budget is not None else DEFAULT_BUDGETS))]
            return run_budget_sweep(
                candidates,
                evaluate,
                seeds=seed_list,
                budgets=budget_list,
                strategies=strategies,
            )
        return compare_strategies(
            candidates,
            evaluate,
            budget=int(budget) if budget is not None else None,
            seed=seed,
            strategies=strategies,
        )
    finally:
        if owns_eval:
            close_loaded(loaded_by_method)


def serialize_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe study dump. Records are summarized, never filled in."""
    skip = {"strategies", "grid"}
    out = {k: v for k, v in study.items() if k not in skip}
    strats = {}
    for name, payload in (study.get("strategies") or {}).items():
        recs = payload.get("records") or []
        strats[name] = {k: v for k, v in payload.items() if k != "records"}
        strats[name]["records"] = [
            r.model_dump() if hasattr(r, "model_dump") else r for r in recs
        ]
    out["strategies"] = strats
    grid = study.get("grid")
    if grid is not None:
        recs = grid.get("records") or []
        out["grid"] = {k: v for k, v in grid.items() if k != "records"}
        out["grid"]["records"] = [
            r.model_dump() if hasattr(r, "model_dump") else r for r in recs
        ]
    return out
