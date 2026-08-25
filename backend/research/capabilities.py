"""Probe which experiments this machine can actually run."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from research.env import collect_environment, select_device


def _try_import(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, f"{name} import ok"
    except Exception as exc:
        return False, f"{name} not available: {exc}"


def probe() -> Dict[str, Any]:
    """Return a capability matrix. Unsupported is a first-class result, not a guess."""
    env = collect_environment()
    device = select_device()
    torch_ok, torch_reason = _try_import("torch")
    transformers_ok, transformers_reason = _try_import("transformers")
    bnb_ok, bnb_reason = _try_import("bitsandbytes")
    gptq_ok, gptq_reason = _try_import("auto_gptq")
    awq_ok, awq_reason = _try_import("awq")
    if not awq_ok:
        awq_ok, awq_reason = _try_import("autoawq")
    llama_ok, llama_reason = _try_import("llama_cpp")
    vllm_ok, vllm_reason = _try_import("vllm")

    cuda = bool(env.get("torch", {}).get("cuda_available"))
    mps = bool(env.get("torch", {}).get("mps_available"))

    def entry(supported: bool, reason: str, backends: List[str]) -> Dict[str, Any]:
        return {
            "supported": supported,
            "reason": reason,
            "backends": backends,
        }

    experiments = {
        "transformers_fp32": entry(
            torch_ok and transformers_ok,
            "ready" if torch_ok and transformers_ok else f"{torch_reason}; {transformers_reason}",
            ["transformers"],
        ),
        "transformers_fp16": entry(
            torch_ok and transformers_ok and device in {"cuda", "mps"},
            "fp16 is measured on CUDA or MPS; CPU stays fp32"
            if torch_ok and transformers_ok and device in {"cuda", "mps"}
            else "fp16 skipped on CPU or missing torch/transformers",
            ["transformers"],
        ),
        "transformers_bf16": entry(
            torch_ok and transformers_ok and cuda,
            "bf16 measured on CUDA when the GPU supports it"
            if cuda
            else "bf16 requires CUDA",
            ["transformers"],
        ),
        "dynamic_int8": entry(
            torch_ok and transformers_ok,
            "PyTorch dynamic int8 on Linear layers (CPU path is the supported one)",
            ["transformers"],
        ),
        "int8_bnb": entry(
            bool(bnb_ok and cuda),
            "bitsandbytes LLM.int8() requires CUDA"
            if not cuda
            else (bnb_reason if not bnb_ok else "ready"),
            ["transformers+bitsandbytes"],
        ),
        "int4_bnb": entry(
            bool(bnb_ok and cuda),
            "bitsandbytes NF4/INT4 requires CUDA"
            if not cuda
            else (bnb_reason if not bnb_ok else "ready"),
            ["transformers+bitsandbytes"],
        ),
        "gptq": entry(
            bool((gptq_ok or transformers_ok) and cuda),
            "GPTQ load/quantize requires CUDA plus auto-gptq or a pre-quantized GPTQ checkpoint"
            if not cuda
            else ("ready to attempt GPTQ load" if gptq_ok or transformers_ok else gptq_reason),
            ["transformers", "auto-gptq"],
        ),
        "awq": entry(
            bool(awq_ok and cuda),
            "AWQ requires CUDA and autoawq (or an AWQ checkpoint loadable by transformers)"
            if not cuda
            else (awq_reason if not awq_ok else "ready"),
            ["autoawq", "transformers"],
        ),
        "gguf": entry(
            llama_ok,
            llama_reason if not llama_ok else "ready if a GGUF file or HF GGUF repo is provided",
            ["llama.cpp"],
        ),
        "vllm": entry(
            bool(vllm_ok and cuda),
            "vLLM is CUDA-only in this platform"
            if not cuda
            else (vllm_reason if not vllm_ok else "ready"),
            ["vllm"],
        ),
        "tensorrt_llm": entry(
            False,
            "TensorRT-LLM is not installed or supported on MacBook / free Colab",
            ["tensorrt_llm"],
        ),
        "smoothquant": entry(
            False,
            "SmoothQuant is not wired: needs a dedicated calibration toolkit not bundled here",
            [],
        ),
        "squeezellm": entry(
            False,
            "SqueezeLLM is not bundled; no silent fallback numbers are produced",
            [],
        ),
        "kv_cache_hf": entry(
            torch_ok and transformers_ok,
            "Hugging Face use_cache / prefix reuse / sliding-window truncation can be measured",
            ["transformers"],
        ),
        "paged_attention": entry(
            bool(vllm_ok and cuda),
            "PagedAttention is measured only via vLLM; HF KV-cache scaling is a separate experiment",
            ["vllm"],
        ),
        "speculative_decoding": entry(
            torch_ok and transformers_ok,
            "Greedy draft-target verification is measured when both models load with a shared vocab",
            ["transformers"],
        ),
        "continuous_batching": entry(
            torch_ok and transformers_ok,
            "Token-level continuous batching vs static batching is measured in-process",
            ["transformers"],
        ),
    }

    return {
        "device": device,
        "cuda": cuda,
        "mps": mps,
        "environment": {
            "system": env.get("system"),
            "machine": env.get("machine"),
            "python": env.get("python"),
            "colab": env.get("colab"),
            "torch": env.get("torch"),
            "packages": env.get("packages"),
        },
        "experiments": experiments,
    }


def is_supported(method: str) -> Tuple[bool, str]:
    matrix = probe()["experiments"]
    if method not in matrix:
        return False, f"unknown experiment/method: {method}"
    item = matrix[method]
    return bool(item["supported"]), str(item["reason"])
