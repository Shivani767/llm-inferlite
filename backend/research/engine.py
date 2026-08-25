"""Wall-clock benchmark engine. Missing hardware/libraries are labeled, never scored."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from research.backends import (
    DEFAULT_PROMPT,
    GenerationSample,
    LoadedModel,
    generate_gguf,
    generate_transformers,
    try_load,
)
from research.capabilities import probe
from research.env import collect_environment, select_device, set_seed, utc_now
from research.memory import snapshot
from research.metrics import mean_or_none, percentile_stats
from research.schema import (
    BenchmarkMetrics,
    ExperimentRecord,
    Status,
    errored,
    unsupported,
)


def _sample_dict(sample: GenerationSample) -> Dict[str, Any]:
    return {
        "prompt_tokens": sample.prompt_tokens,
        "completion_tokens": sample.completion_tokens,
        "ttft_ms": sample.ttft_ms,
        "e2e_ms": sample.e2e_ms,
        "inter_token_ms_mean": mean_or_none(sample.inter_token_ms),
        "tokens_per_sec": sample.tokens_per_sec,
        "text_preview": (sample.text or "")[:200],
        "memory": sample.memory,
    }


def _metrics_from_samples(
    samples: List[GenerationSample],
    load_time_s: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> BenchmarkMetrics:
    rss_peaks = []
    gpu_alloc = []
    gpu_reserved = []
    for s in samples:
        after = (s.memory or {}).get("after") or {}
        if after.get("rss_mb") is not None:
            rss_peaks.append(after["rss_mb"])
        if after.get("max_allocated_mb") is not None:
            gpu_alloc.append(after["max_allocated_mb"])
        if after.get("reserved_mb") is not None:
            gpu_reserved.append(after["reserved_mb"])
        elif after.get("max_allocated_mb") is None and after.get("allocated_mb") is not None:
            gpu_alloc.append(after["allocated_mb"])

    prompt_tokens = samples[0].prompt_tokens if samples else None
    completion_mean = mean_or_none([s.completion_tokens for s in samples])
    return BenchmarkMetrics(
        load_time_s=load_time_s,
        ttft_ms=percentile_stats([s.ttft_ms for s in samples]),
        e2e_latency_ms=percentile_stats([s.e2e_ms for s in samples]),
        inter_token_latency_ms=percentile_stats(
            [x for s in samples for x in s.inter_token_ms]
        ),
        tokens_per_sec=percentile_stats(
            [s.tokens_per_sec for s in samples if s.tokens_per_sec is not None]
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens_mean=completion_mean,
        peak_rss_mb=max(rss_peaks) if rss_peaks else None,
        peak_gpu_allocated_mb=max(gpu_alloc) if gpu_alloc else None,
        peak_gpu_reserved_mb=max(gpu_reserved) if gpu_reserved else None,
        extra=extra or {},
    )


def _generate(loaded: LoadedModel, prompt: str, max_new_tokens: int) -> GenerationSample:
    if loaded.backend == "llama.cpp":
        return generate_gguf(loaded, prompt, max_new_tokens=max_new_tokens)
    return generate_transformers(loaded, prompt, max_new_tokens=max_new_tokens)


def run_benchmark(
    *,
    model_id: str,
    method: str = "fp32",
    backend: str = "transformers",
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 32,
    warmup_runs: int = 1,
    measure_runs: int = 3,
    seed: int = 42,
    device: Optional[str] = None,
    experiment_type: str = "benchmark",
    config: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, Any]] = None,
    loaded: Optional[LoadedModel] = None,
    **load_kwargs: Any,
) -> ExperimentRecord:
    """
    Warm up, then measure TTFT / TPS / latency percentiles / memory / load time.

    Returns status=unsupported or error instead of invented numbers.
    """
    device = device or select_device()
    env = environment or collect_environment(seed=seed)
    cfg = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "seed": seed,
        **(config or {}),
        **{k: v for k, v in load_kwargs.items() if k not in {"model", "tokenizer"}},
    }
    exp_id = f"{utc_now().replace(':', '')}_{method}_{uuid.uuid4().hex[:8]}"
    set_seed(seed)

    method_key = method.lower()
    if backend == "llama.cpp":
        method_key = "gguf"

    caps = probe()["experiments"]
    cap_name = {
        "fp32": "transformers_fp32",
        "fp16": "transformers_fp16",
        "bf16": "transformers_bf16",
        "dynamic_int8": "dynamic_int8",
        "int8_bnb": "int8_bnb",
        "bitsandbytes_int8": "int8_bnb",
        "int4_bnb": "int4_bnb",
        "bitsandbytes_int4": "int4_bnb",
        "nf4": "int4_bnb",
        "gptq": "gptq",
        "awq": "awq",
        "gguf": "gguf",
        "gguf_q4_k_m": "gguf",
        "vllm": "vllm",
        "tensorrt_llm": "tensorrt_llm",
        "smooth_quant": "smoothquant",
        "squeeze_llm": "squeezellm",
    }.get(method_key, method_key)

    if loaded is None and cap_name in caps and not caps[cap_name]["supported"]:
        return unsupported(
            experiment_id=exp_id,
            experiment_type=experiment_type,
            model_id=model_id,
            backend=backend,
            method=method,
            device=device,
            precision=method,
            reason=caps[cap_name]["reason"],
            environment=env,
            config=cfg,
        )

    owns_model = loaded is None
    if loaded is None:
        loaded, err = try_load(method_key, model_id, device=device, **load_kwargs)
        if err or loaded is None:
            return unsupported(
                experiment_id=exp_id,
                experiment_type=experiment_type,
                model_id=model_id,
                backend=backend,
                method=method,
                device=device,
                precision=method,
                reason=err or "load returned no model",
                environment=env,
                config=cfg,
            )

    try:
        for _ in range(max(0, warmup_runs)):
            _generate(loaded, prompt, max_new_tokens)
        samples: List[GenerationSample] = []
        for _ in range(max(1, measure_runs)):
            samples.append(_generate(loaded, prompt, max_new_tokens))
        extra = {
            "weight_mb": (loaded.extras or {}).get("weight_mb"),
            "backend_extras": {k: v for k, v in (loaded.extras or {}).items() if k != "weight_mb"},
            "memory_end": snapshot(),
        }
        metrics = _metrics_from_samples(samples, loaded.load_time_s, extra=extra)
        metrics.model_weight_mb = extra.get("weight_mb")
        notes = [
            "All timings are wall-clock on this machine.",
            "First-token time is prefill latency of the decode loop.",
        ]
        if warmup_runs:
            notes.append(f"Discarded {warmup_runs} warmup run(s) before measurement.")
        return ExperimentRecord(
            experiment_id=exp_id,
            experiment_type=experiment_type,
            status=Status.MEASURED,
            timestamp_utc=utc_now(),
            model_id=loaded.model_id,
            backend=loaded.backend,
            method=loaded.method,
            device=loaded.device,
            precision=loaded.precision,
            config=cfg,
            environment=env,
            metrics=metrics,
            samples=[_sample_dict(s) for s in samples],
            notes=notes,
        )
    except Exception as exc:
        return errored(
            experiment_id=exp_id,
            experiment_type=experiment_type,
            model_id=model_id,
            backend=backend,
            method=method,
            device=device,
            precision=method,
            reason=f"{type(exc).__name__}: {exc}",
            environment=env,
            config=cfg,
        )
    finally:
        if owns_model and loaded is not None:
            loaded.close()
