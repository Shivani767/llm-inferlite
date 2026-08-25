"""Static vs continuous (token-level) batching. Measures real wall-clock throughput and latency."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from research.backends import load_transformers
from research.env import collect_environment, select_device, set_seed, utc_now
from research.memory import snapshot
from research.metrics import percentile_stats
from research.schema import (
    BenchmarkMetrics,
    ExperimentRecord,
    Status,
    errored,
    unsupported,
)

PROMPTS = [
    "Explain KV cache in one sentence.",
    "What is quantization?",
    "Define time to first token.",
    "Why is decode memory-bandwidth bound?",
    "What is speculative decoding?",
    "Name one GGUF quantization type.",
    "What does P50 latency mean?",
    "Why record environment metadata?",
]


def _greedy_step(model, input_ids, past=None):
    import torch

    with torch.no_grad():
        if past is None:
            out = model(input_ids=input_ids, use_cache=True)
        else:
            out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        return nxt, out.past_key_values


def _static_batch(
    model, tokenizer, prompts: List[str], max_new: int, device: str
) -> Dict[str, Any]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    t0 = time.perf_counter()
    latencies = []
    # Generate sequentially per request inside one padded batch of decode steps
    # using generate() for a true static batch (all sequences start together).
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    e2e = (time.perf_counter() - t0) * 1000.0
    new_tokens = int(out.shape[1] - input_ids.shape[1]) * int(out.shape[0])
    for _ in prompts:
        latencies.append(e2e)
    return {
        "e2e_ms": e2e,
        "per_request_latency_ms": latencies,
        "completion_tokens": new_tokens,
        "tokens_per_sec": (new_tokens / (e2e / 1000.0)) if e2e > 0 else None,
        "memory": snapshot(),
    }


def _continuous_batch(
    model, tokenizer, prompts: List[str], max_new: int, device: str, max_batch: int
) -> Dict[str, Any]:
    """
    In-process continuous batching: requests join as slots free.
    Arrival is immediate (all queued at t=0) so the comparison isolates
    decode-step packing vs waiting for a full static batch to finish.
    """
    import torch

    encoded_list = [tokenizer(p, return_tensors="pt")["input_ids"].to(device) for p in prompts]
    pending = list(range(len(prompts)))
    active: Dict[int, Dict[str, Any]] = {}
    finished_latency: Dict[int, float] = {}
    total_new = 0
    t0 = time.perf_counter()

    def admit():
        while pending and len(active) < max_batch:
            i = pending.pop(0)
            ids = encoded_list[i]
            nxt, past = _greedy_step(model, ids, None)
            active[i] = {
                "past": past,
                "token": nxt,
                "steps": 1,
                "start": t0,
            }

    admit()
    while active:
        # one decode step for every active sequence (separate forwards; packing is the scheduler)
        finished = []
        for i, st in list(active.items()):
            nxt, past = _greedy_step(model, st["token"], st["past"])
            st["past"] = past
            st["token"] = nxt
            st["steps"] += 1
            total_new += 1
            eos = getattr(tokenizer, "eos_token_id", None)
            if st["steps"] >= max_new or (eos is not None and int(nxt.item()) == eos):
                finished_latency[i] = (time.perf_counter() - t0) * 1000.0
                finished.append(i)
        for i in finished:
            active.pop(i, None)
        admit()

    e2e = (time.perf_counter() - t0) * 1000.0
    lats = [finished_latency[i] for i in sorted(finished_latency)]
    return {
        "e2e_ms": e2e,
        "per_request_latency_ms": lats,
        "completion_tokens": total_new,
        "tokens_per_sec": (total_new / (e2e / 1000.0)) if e2e > 0 else None,
        "memory": snapshot(),
    }


def run_batching_suite(
    model_id: str,
    *,
    num_requests: int = 4,
    max_batch_size: int = 2,
    max_new_tokens: int = 8,
    seed: int = 42,
    model: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> List[ExperimentRecord]:
    env = collect_environment(seed=seed)
    device = select_device()
    set_seed(seed)
    prompts = (PROMPTS * ((num_requests // len(PROMPTS)) + 1))[:num_requests]

    try:
        loaded = load_transformers(
            model_id,
            method="fp32" if device == "cpu" else "fp16",
            device=device,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
    except Exception as exc:
        return [
            unsupported(
                experiment_id=f"batch_{uuid.uuid4().hex[:8]}",
                experiment_type="continuous_batching",
                model_id=model_id,
                backend="transformers",
                method="batching",
                device=device,
                precision="fp32",
                reason=f"could not load model: {type(exc).__name__}: {exc}",
                environment=env,
                config={"num_requests": num_requests, "max_batch_size": max_batch_size},
            )
        ]

    tok = loaded._tokenizer
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    results: List[ExperimentRecord] = []

    for name, fn, extra_cfg in (
        (
            "static_batch",
            lambda: _static_batch(loaded._impl, tok, prompts, max_new_tokens, loaded.device),
            {},
        ),
        (
            "continuous_batch",
            lambda: _continuous_batch(
                loaded._impl, tok, prompts, max_new_tokens, loaded.device, max_batch_size
            ),
            {"max_batch_size": max_batch_size},
        ),
    ):
        exp_id = f"batch_{name}_{uuid.uuid4().hex[:8]}"
        try:
            out = fn()
            lats = out["per_request_latency_ms"]
            results.append(
                ExperimentRecord(
                    experiment_id=exp_id,
                    experiment_type="continuous_batching",
                    status=Status.MEASURED,
                    timestamp_utc=utc_now(),
                    model_id=loaded.model_id,
                    backend="transformers",
                    method=name,
                    device=loaded.device,
                    precision=loaded.precision,
                    config={
                        "num_requests": num_requests,
                        "max_new_tokens": max_new_tokens,
                        **extra_cfg,
                    },
                    environment=env,
                    metrics=BenchmarkMetrics(
                        load_time_s=loaded.load_time_s,
                        e2e_latency_ms=percentile_stats(lats),
                        tokens_per_sec=percentile_stats(
                            [out["tokens_per_sec"]] if out["tokens_per_sec"] else []
                        ),
                        extra={
                            "batch_wall_clock_ms": out["e2e_ms"],
                            "completion_tokens": out["completion_tokens"],
                            "memory": out["memory"],
                        },
                    ),
                    samples=[
                        {"request_index": i, "latency_ms": lat} for i, lat in enumerate(lats)
                    ],
                    notes=[
                        "Static batch starts every request together via model.generate().",
                        "Continuous batch admits new requests as slots free (in-process scheduler).",
                        "This is not vLLM continuous batching; it is a research implementation.",
                    ],
                )
            )
        except Exception as exc:
            results.append(
                errored(
                    experiment_id=exp_id,
                    experiment_type="continuous_batching",
                    model_id=model_id,
                    backend="transformers",
                    method=name,
                    device=device,
                    precision=loaded.precision,
                    reason=f"{type(exc).__name__}: {exc}",
                    environment=env,
                    config={"num_requests": num_requests},
                )
            )
    loaded.close()
    return results
