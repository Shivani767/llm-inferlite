"""Latency and throughput statistics. Never invents missing samples."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from research.schema import PercentileStats


def percentile_stats(values: Sequence[float]) -> Optional[PercentileStats]:
    """Return percentile stats, or None when there are no finite samples."""
    clean: List[float] = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float64)
    return PercentileStats(
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        min=float(arr.min()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(arr.max()),
    )


def tokens_per_second(num_tokens: int, elapsed_s: float) -> Optional[float]:
    if num_tokens <= 0 or elapsed_s <= 0:
        return None
    return float(num_tokens) / float(elapsed_s)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.mean(clean))
