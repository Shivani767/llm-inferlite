"""Process and accelerator memory snapshots."""

from __future__ import annotations

from typing import Any, Dict, Optional


def rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024**2), 3)
    except Exception:
        return None


def gpu_memory_mb() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "allocated_mb": None,
        "reserved_mb": None,
        "max_allocated_mb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        out["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 3)
        out["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 3)
        out["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 3)
    except Exception:
        return out
    return out


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def snapshot() -> Dict[str, Any]:
    gpu = gpu_memory_mb()
    return {"rss_mb": rss_mb(), **gpu}
