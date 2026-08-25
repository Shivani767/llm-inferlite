"""Latency and throughput statistics. Never invents missing samples."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from research.schema import PercentileStats

# Two-sided 97.5% Student-t critical values, keyed by degrees of freedom (n-1).
# Avoids a scipy dependency for n=5 seed studies.
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_crit_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_CRIT_975:
        return _T_CRIT_975[df]
    if df > 120:
        return 1.96
    keys = sorted(_T_CRIT_975)
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T_CRIT_975[lo]
    w = (df - lo) / (hi - lo)
    return _T_CRIT_975[lo] * (1.0 - w) + _T_CRIT_975[hi] * w


def mean_std_ci95(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Sample mean, unbiased std, and 95% t-interval. Empty input → nulls, not zeros."""
    arr = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(v)],
        dtype=np.float64,
    )
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(arr.std(ddof=1))
    se = std / np.sqrt(n)
    tcrit = _t_crit_975(n - 1)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
    }


def percentile_stats(values: Sequence[float]) -> Optional[PercentileStats]:
    """Return percentile stats, or None when there are no finite samples."""
    clean: List[float] = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float64)
    mean = float(arr.mean())
    n = int(arr.size)
    ci95_low = None
    ci95_high = None
    if n >= 2:
        se = float(arr.std(ddof=1) / np.sqrt(n))
        ci95_low = mean - 1.96 * se
        ci95_high = mean + 1.96 * se
    return PercentileStats(
        n=n,
        mean=mean,
        std=float(arr.std(ddof=0)),
        min=float(arr.min()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(arr.max()),
        ci95_low=ci95_low,
        ci95_high=ci95_high,
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
