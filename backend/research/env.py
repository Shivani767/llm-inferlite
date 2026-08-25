"""Capture hardware/software metadata for reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def _git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        return None
    return None


def _cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"installed": False}
    try:
        import torch
    except Exception as exc:
        info["import_error"] = str(exc)
        return info

    info.update(
        {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_built": bool(getattr(torch.version, "cuda", None)),
            "cuda_version": getattr(torch.version, "cuda", None),
            "mps_built": bool(getattr(getattr(torch.backends, "mps", None), "is_built", lambda: False)()),
        }
    )
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False
    try:
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps.is_available()) if mps is not None else False
    except Exception:
        info["mps_available"] = False

    if info.get("cuda_available"):
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu"] = {
                "index": int(idx),
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
                "multi_processor_count": int(props.multi_processor_count),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        except Exception as exc:
            info["gpu_error"] = str(exc)
    return info


def select_device(prefer: Optional[str] = None) -> str:
    """Pick the best available torch device. Does not invent accelerators."""
    if prefer and prefer not in {"auto", ""}:
        return prefer
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def collect_environment(seed: Optional[int] = None) -> Dict[str, Any]:
    torch_info = _torch_info()
    env: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": _cpu_brand(),
        "cpu_count_logical": os.cpu_count(),
        "seed": seed,
        "git_commit": _git_commit(),
        "packages": {
            "torch": torch_info.get("version"),
            "transformers": _pkg_version("transformers"),
            "accelerate": _pkg_version("accelerate"),
            "bitsandbytes": _pkg_version("bitsandbytes"),
            "auto_gptq": _pkg_version("auto_gptq"),
            "autoawq": _pkg_version("autoawq"),
            "llama_cpp": _pkg_version("llama_cpp"),
            "vllm": _pkg_version("vllm"),
            "numpy": _pkg_version("numpy"),
        },
        "torch": torch_info,
        "colab": bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")),
        "hf_home": os.environ.get("HF_HOME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        env["ram"] = {
            "total_mb": round(vm.total / (1024**2), 2),
            "available_mb": round(vm.available / (1024**2), 2),
        }
    except Exception:
        env["ram"] = None
    return env


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
