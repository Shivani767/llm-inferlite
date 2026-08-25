"""Energy sampling. Missing sensors are unsupported — never guessed watts."""

from __future__ import annotations

from typing import Any, Dict, Optional


def probe_energy() -> Dict[str, Any]:
    """Return whether a real power sensor is available."""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                handle = None
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    return {
                        "supported": True,
                        "backend": "pynvml",
                        "instant_watts": float(mw) / 1000.0,
                    }
                except Exception as exc:
                    return {
                        "supported": False,
                        "reason": f"CUDA present but NVML power unavailable: {exc}",
                    }
            except Exception as exc:
                return {"supported": False, "reason": str(exc)}
    except Exception:
        pass
    return {
        "supported": False,
        "reason": "no NVML/power sensor on this machine; energy is not scored",
    }


def joules_from_watts(watts: Optional[float], elapsed_s: Optional[float]) -> Optional[float]:
    if watts is None or elapsed_s is None or watts <= 0 or elapsed_s <= 0:
        return None
    return float(watts) * float(elapsed_s)
