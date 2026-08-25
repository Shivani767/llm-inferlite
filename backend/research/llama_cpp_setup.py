"""Install llama-cpp-python from a prebuilt wheel. Never compile from source on Colab."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, List, Optional


CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/{tag}"
CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
# py3-none wheel: large (~2 GB) but avoids a 30–40 min Colab source build.
GITHUB_CUDA_WHEEL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v0.3.34-cu121/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"
)


def cuda_wheel_tags() -> List[str]:
    tags: List[str] = []
    try:
        import torch

        raw = getattr(torch.version, "cuda", None) or ""
        parts = raw.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            tags.append(f"cu{parts[0]}{parts[1]}")
    except Exception:
        pass
    for extra in ("cu124", "cu125", "cu121", "cu122", "cu118"):
        if extra not in tags:
            tags.append(extra)
    return tags


def _pip(args: List[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pip", *args])


def _import_llama() -> Optional[str]:
    try:
        import llama_cpp

        return getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        return None


def already_imported() -> Optional[str]:
    return _import_llama()


def install_commands() -> List[List[str]]:
    """pip argv lists. Every CUDA/CPU attempt is `--only-binary=:all:` (no sdist compile)."""
    cmds: List[List[str]] = []
    common = ["install", "-q", "llama-cpp-python", "--only-binary=:all:", "--no-cache-dir"]
    for tag in cuda_wheel_tags():
        cmds.append([*common, "--extra-index-url", CUDA_INDEX.format(tag=tag)])
    cmds.append([*common, "--extra-index-url", CPU_INDEX])
    cmds.append(["install", "-q", "--no-cache-dir", GITHUB_CUDA_WHEEL])
    return cmds


def ensure_llama_cpp(*, dry_run: bool = False) -> Dict[str, Any]:
    """Return import status. Does not invent GGUF timings."""
    version = _import_llama()
    if version:
        return {"ok": True, "version": version, "action": "already_present"}
    if dry_run:
        return {"ok": False, "version": None, "action": "dry_run", "commands": install_commands()}

    tried: List[str] = []
    for args in install_commands():
        label = " ".join(args)
        tried.append(label)
        _pip(args)
        version = _import_llama()
        if version:
            return {"ok": True, "version": version, "action": "installed", "pip": label}
    return {
        "ok": False,
        "version": None,
        "action": "failed",
        "reason": "no prebuilt llama-cpp-python wheel imported; refusing to compile from source",
        "tried": tried,
    }
