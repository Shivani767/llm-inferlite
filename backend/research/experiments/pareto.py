"""Pareto front over measured experiments only."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from research.schema import ExperimentRecord, Status


def _value(record: ExperimentRecord, key: str) -> Optional[float]:
    m = record.metrics
    if m is None:
        return None
    if key == "tokens_per_sec":
        return m.tokens_per_sec.mean if m.tokens_per_sec else None
    if key == "p50_latency_ms":
        src = m.e2e_latency_ms
        return src.p50 if src else None
    if key == "p95_latency_ms":
        src = m.e2e_latency_ms
        return src.p95 if src else None
    if key == "p99_latency_ms":
        src = m.e2e_latency_ms
        return src.p99 if src else None
    if key == "ttft_ms":
        return m.ttft_ms.mean if m.ttft_ms else None
    if key == "memory_mb":
        if m.peak_gpu_allocated_mb is not None:
            return m.peak_gpu_allocated_mb
        return m.peak_rss_mb
    if key == "load_time_s":
        return m.load_time_s
    if key == "perplexity":
        return m.perplexity
    extra = m.extra or {}
    if key in extra and extra[key] is not None:
        try:
            return float(extra[key])
        except (TypeError, ValueError):
            return None
    return None


def dominates(
    a: Sequence[float],
    b: Sequence[float],
) -> bool:
    """True if a is better-or-equal on all objectives and strictly better on one.
    All objectives are already oriented so that smaller is better.
    """
    le = all(x <= y for x, y in zip(a, b))
    lt = any(x < y for x, y in zip(a, b))
    return le and lt


def pareto_front(
    records: Iterable[ExperimentRecord],
    *,
    minimize: Optional[List[str]] = None,
    maximize: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute the Pareto front from measured records.

    Default: minimize p95 latency and memory, maximize tokens/sec.
    Unsupported/error rows are excluded.
    """
    minimize = minimize or ["p95_latency_ms", "memory_mb"]
    maximize = maximize or ["tokens_per_sec"]

    points: List[Tuple[ExperimentRecord, Tuple[float, ...], Dict[str, Optional[float]]]] = []
    for rec in records:
        if rec.status != Status.MEASURED:
            continue
        oriented: List[float] = []
        raw: Dict[str, Optional[float]] = {}
        skip = False
        for key in minimize:
            val = _value(rec, key)
            raw[key] = val
            if val is None:
                skip = True
                break
            oriented.append(val)
        if skip:
            continue
        for key in maximize:
            val = _value(rec, key)
            raw[key] = val
            if val is None:
                skip = True
                break
            oriented.append(-val)
        if skip:
            continue
        points.append((rec, tuple(oriented), raw))

    front: List[Dict[str, Any]] = []
    for i, (rec_i, vec_i, raw_i) in enumerate(points):
        dominated = False
        for j, (rec_j, vec_j, _) in enumerate(points):
            if i == j:
                continue
            if dominates(vec_j, vec_i):
                dominated = True
                break
        if not dominated:
            front.append(
                {
                    "experiment_id": rec_i.experiment_id,
                    "method": rec_i.method,
                    "backend": rec_i.backend,
                    "model_id": rec_i.model_id,
                    "device": rec_i.device,
                    "objectives": raw_i,
                    "record": rec_i.model_dump(),
                }
            )
    return front
