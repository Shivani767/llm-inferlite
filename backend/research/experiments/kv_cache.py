"""KV-cache scaling: real memory and latency vs context length. No invented hit rates."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from research.backends import LoadedModel, generate_transformers, load_transformers
from research.env import collect_environment, select_device, set_seed, utc_now
from research.metrics import percentile_stats
from research.schema import (
    BenchmarkMetrics,
    ExperimentRecord,
    Status,
    errored,
    unsupported,
)


def _pad_prompt(tokenizer, base: str, target_tokens: int, device: str) -> str:
    """Build a prompt with approximately target_tokens using repeated filler."""
    filler = " measurement token"
    text = base
    # Grow then trim by token count when a real tokenizer exists.
    while True:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        n = int(ids.shape[1])
        if n >= target_tokens:
            trimmed = ids[:, :target_tokens]
            try:
                return tokenizer.decode(trimmed[0], skip_special_tokens=False)
            except Exception:
                return text
        text += filler * max(1, (target_tokens - n))


def run_kv_cache_suite(
    model_id: str,
    *,
    context_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 16,
    warmup_runs: int = 0,
    measure_runs: int = 2,
    seed: int = 42,
    strategies: Optional[List[str]] = None,
    model: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> List[ExperimentRecord]:
    """
    Measure prefill TTFT, decode TPS, and RSS/GPU memory as context grows.

    Strategies:
      - dynamic: Hugging Face use_cache=True (standard KV cache)
      - no_cache: use_cache=False (recompute), when it runs
      - sliding_window: truncate prompt to a fixed window (true truncation, not a kernel)
      - prefix: reuse past_key_values for a shared prefix (real cache reuse)
      - paged_attention: unsupported unless vLLM is present (labeled, not faked)
    """
    context_lengths = context_lengths or [32, 64, 128]
    strategies = strategies or ["dynamic", "no_cache", "sliding_window", "prefix", "paged_attention"]
    env = collect_environment(seed=seed)
    device = select_device()
    set_seed(seed)
    results: List[ExperimentRecord] = []

    loaded: Optional[LoadedModel] = None
    try:
        loaded = load_transformers(
            model_id, method="fp32" if device == "cpu" else "fp16",
            device=device, model=model, tokenizer=tokenizer, **kwargs
        )
    except Exception as exc:
        for strat in strategies:
            results.append(
                unsupported(
                    experiment_id=f"kv_{strat}_{uuid.uuid4().hex[:8]}",
                    experiment_type="kv_cache",
                    model_id=model_id,
                    backend="transformers",
                    method=strat,
                    device=device,
                    precision="fp32",
                    reason=f"could not load model: {type(exc).__name__}: {exc}",
                    environment=env,
                    config={"context_lengths": context_lengths},
                )
            )
        return results

    tok = loaded._tokenizer
    window = min(context_lengths) if context_lengths else 32

    for strat in strategies:
        if strat == "paged_attention":
            try:
                import vllm  # noqa: F401
            except Exception:
                results.append(
                    unsupported(
                        experiment_id=f"kv_paged_{uuid.uuid4().hex[:8]}",
                        experiment_type="kv_cache",
                        model_id=model_id,
                        backend="vllm",
                        method="paged_attention",
                        device=device,
                        precision=loaded.precision,
                        reason="PagedAttention is only measured with vLLM, which is not available here",
                        environment=env,
                        config={"context_lengths": context_lengths},
                    )
                )
                continue

        for ctx in context_lengths:
            exp_id = f"kv_{strat}_c{ctx}_{uuid.uuid4().hex[:8]}"
            cfg: Dict[str, Any] = {
                "strategy": strat,
                "context_length": ctx,
                "max_new_tokens": max_new_tokens,
                "sliding_window": window,
            }
            try:
                prompt = _pad_prompt(tok, "Context scaling study.", ctx, loaded.device)
                samples = []
                if strat == "prefix":
                    # Shared prefix measured by running two generations that share a prefix.
                    prefix = prompt
                    suffix = " Continue."
                    # First call builds cache; second call reuses past_key_values.
                    import torch

                    encoded = tok(prefix, return_tensors="pt")
                    encoded = {k: v.to(loaded.device) for k, v in encoded.items()}
                    for i in range(max(1, measure_runs)):
                        t_samples = []
                        with torch.no_grad():
                            import time as _time

                            t0 = _time.perf_counter()
                            out = loaded._impl(**encoded, use_cache=True)
                            prefix_ms = (_time.perf_counter() - t0) * 1000.0
                            past = out.past_key_values
                            suffix_ids = tok(suffix, return_tensors="pt")["input_ids"].to(loaded.device)
                            t1 = _time.perf_counter()
                            out2 = loaded._impl(input_ids=suffix_ids, past_key_values=past, use_cache=True)
                            reused_ms = (_time.perf_counter() - t1) * 1000.0
                        from research.backends import GenerationSample
                        from research.memory import snapshot

                        t_samples.append(
                            GenerationSample(
                                prompt_tokens=int(encoded["input_ids"].shape[1]),
                                completion_tokens=1,
                                ttft_ms=reused_ms,
                                e2e_ms=prefix_ms + reused_ms,
                                inter_token_ms=[],
                                tokens_per_sec=None,
                                text="",
                                memory={"after": snapshot(), "prefix_prefill_ms": prefix_ms},
                            )
                        )
                        samples.extend(t_samples)
                    metrics = BenchmarkMetrics(
                        load_time_s=loaded.load_time_s,
                        ttft_ms=percentile_stats([s.ttft_ms for s in samples]),
                        e2e_latency_ms=percentile_stats([s.e2e_ms for s in samples]),
                        peak_rss_mb=max(
                            (s.memory.get("after") or {}).get("rss_mb") or 0 for s in samples
                        )
                        or None,
                        peak_gpu_allocated_mb=max(
                            (s.memory.get("after") or {}).get("max_allocated_mb") or 0 for s in samples
                        )
                        or None,
                        extra={
                            "prefix_prefill_ms_mean": sum(
                                (s.memory.get("prefix_prefill_ms") or 0) for s in samples
                            )
                            / len(samples),
                            "reuse_ttft_ms_mean": sum(s.ttft_ms for s in samples) / len(samples),
                        },
                    )
                    results.append(
                        ExperimentRecord(
                            experiment_id=exp_id,
                            experiment_type="kv_cache",
                            status=Status.MEASURED,
                            timestamp_utc=utc_now(),
                            model_id=loaded.model_id,
                            backend="transformers",
                            method="prefix",
                            device=loaded.device,
                            precision=loaded.precision,
                            config=cfg,
                            environment=env,
                            metrics=metrics,
                            samples=[
                                {
                                    "ttft_ms": s.ttft_ms,
                                    "e2e_ms": s.e2e_ms,
                                    "prompt_tokens": s.prompt_tokens,
                                    "memory": s.memory,
                                }
                                for s in samples
                            ],
                            notes=[
                                "Prefix reuse measures a second forward that consumes past_key_values.",
                                "This is not vLLM prefix caching; it is HF cache reuse.",
                            ],
                        )
                    )
                    continue

                use_cache = strat != "no_cache"
                run_prompt = prompt
                if strat == "sliding_window":
                    run_prompt = _pad_prompt(tok, "Context scaling study.", min(ctx, window), loaded.device)
                    cfg["effective_tokens"] = min(ctx, window)
                    cfg["note"] = "Sliding window here is prompt truncation to a fixed token window."

                for _ in range(max(0, warmup_runs)):
                    generate_transformers(
                        loaded, run_prompt, max_new_tokens=max_new_tokens, use_cache=use_cache
                    )
                from research.backends import GenerationSample as GS

                gens = []
                for _ in range(max(1, measure_runs)):
                    gens.append(
                        generate_transformers(
                            loaded, run_prompt, max_new_tokens=max_new_tokens, use_cache=use_cache
                        )
                    )
                rss = [
                    (g.memory.get("after") or {}).get("rss_mb")
                    for g in gens
                    if (g.memory.get("after") or {}).get("rss_mb") is not None
                ]
                gpu = [
                    (g.memory.get("after") or {}).get("max_allocated_mb")
                    or (g.memory.get("after") or {}).get("allocated_mb")
                    for g in gens
                ]
                gpu = [x for x in gpu if x is not None]
                metrics = BenchmarkMetrics(
                    load_time_s=loaded.load_time_s,
                    ttft_ms=percentile_stats([g.ttft_ms for g in gens]),
                    e2e_latency_ms=percentile_stats([g.e2e_ms for g in gens]),
                    inter_token_latency_ms=percentile_stats(
                        [x for g in gens for x in g.inter_token_ms]
                    ),
                    tokens_per_sec=percentile_stats(
                        [g.tokens_per_sec for g in gens if g.tokens_per_sec]
                    ),
                    prompt_tokens=gens[0].prompt_tokens,
                    completion_tokens_mean=sum(g.completion_tokens for g in gens) / len(gens),
                    peak_rss_mb=max(rss) if rss else None,
                    peak_gpu_allocated_mb=max(gpu) if gpu else None,
                )
                notes = [
                    f"Strategy={strat}; requested context tokens={ctx}.",
                    "Memory is process RSS and CUDA allocated bytes when CUDA is present.",
                ]
                if strat == "sliding_window":
                    notes.append("Not a fused sliding-window attention kernel; prompt is truncated.")
                results.append(
                    ExperimentRecord(
                        experiment_id=exp_id,
                        experiment_type="kv_cache",
                        status=Status.MEASURED,
                        timestamp_utc=utc_now(),
                        model_id=loaded.model_id,
                        backend="transformers",
                        method=strat,
                        device=loaded.device,
                        precision=loaded.precision,
                        config=cfg,
                        environment=env,
                        metrics=metrics,
                        samples=[
                            {
                                "ttft_ms": g.ttft_ms,
                                "e2e_ms": g.e2e_ms,
                                "tokens_per_sec": g.tokens_per_sec,
                                "prompt_tokens": g.prompt_tokens,
                                "completion_tokens": g.completion_tokens,
                                "memory": g.memory,
                            }
                            for g in gens
                        ],
                        notes=notes,
                    )
                )
            except Exception as exc:
                results.append(
                    errored(
                        experiment_id=exp_id,
                        experiment_type="kv_cache",
                        model_id=model_id,
                        backend="transformers",
                        method=strat,
                        device=device,
                        precision=loaded.precision,
                        reason=f"{type(exc).__name__}: {exc}",
                        environment=env,
                        config=cfg,
                    )
                )
    if loaded is not None:
        loaded.close()
    return results
