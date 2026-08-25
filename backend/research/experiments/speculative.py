"""Greedy speculative decoding with a draft model. Measures real acceptance and wall-clock speedup."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from research.backends import LoadedModel, generate_transformers, load_transformers
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


def _vocab_compatible(a, b) -> Tuple[bool, str]:
    va = getattr(a, "vocab_size", None) or getattr(getattr(a, "config", None), "vocab_size", None)
    vb = getattr(b, "vocab_size", None) or getattr(getattr(b, "config", None), "vocab_size", None)
    if va is None or vb is None:
        return True, "vocab sizes not advertised; proceeding"
    if int(va) != int(vb):
        return False, f"vocab mismatch: target={va} draft={vb}"
    return True, "shared vocab size"


def speculative_generate(
    target: LoadedModel,
    draft: LoadedModel,
    prompt: str,
    *,
    gamma: int = 4,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """
    Greedy speculative decoding (Leviathan et al. / Chen et al. style):
    draft proposes `gamma` tokens; target verifies in one forward over the window.
    """
    import torch

    tok = target._tokenizer
    device = target.device
    encoded = tok(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    eos_id = getattr(tok, "eos_token_id", None)

    drafted = 0
    accepted = 0
    generated: List[int] = []
    t0 = time.perf_counter()
    first_token_ms = None

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            prefix = torch.cat(
                [input_ids, torch.tensor([generated], device=device)], dim=1
            ) if generated else input_ids

            draft_tokens: List[int] = []
            draft_cur = prefix
            for _ in range(gamma):
                dout = draft._impl(input_ids=draft_cur)
                nxt = int(torch.argmax(dout.logits[:, -1, :], dim=-1).item())
                draft_tokens.append(nxt)
                draft_cur = torch.cat(
                    [draft_cur, torch.tensor([[nxt]], device=device)], dim=1
                )
            drafted += len(draft_tokens)

            verify = torch.cat(
                [prefix, torch.tensor([draft_tokens], device=device)], dim=1
            )
            tout = target._impl(input_ids=verify)
            # logits[i] predicts token at position i+1
            n_accept = 0
            prefix_len = prefix.shape[1]
            for i, dtok in enumerate(draft_tokens):
                pred = int(torch.argmax(tout.logits[:, prefix_len + i - 1, :], dim=-1).item())
                if pred == dtok:
                    n_accept += 1
                else:
                    break
            accepted += n_accept

            if n_accept == len(draft_tokens):
                bonus = int(torch.argmax(tout.logits[:, -1, :], dim=-1).item())
                new_ids = draft_tokens + [bonus]
            else:
                # correction token from target at the mismatch index
                logits_idx = prefix_len + n_accept - 1
                correction = int(torch.argmax(tout.logits[:, logits_idx, :], dim=-1).item())
                new_ids = draft_tokens[:n_accept] + [correction]

            if first_token_ms is None and new_ids:
                first_token_ms = (time.perf_counter() - t0) * 1000.0

            stop = False
            for tid in new_ids:
                generated.append(tid)
                if eos_id is not None and tid == eos_id:
                    stop = True
                    break
                if len(generated) >= max_new_tokens:
                    stop = True
                    break
            if stop:
                break

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    n = len(generated)
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "completion_tokens": n,
        "ttft_ms": first_token_ms,
        "e2e_ms": e2e_ms,
        "tokens_per_sec": (n / (e2e_ms / 1000.0)) if e2e_ms > 0 else None,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": (accepted / drafted) if drafted else None,
        "gamma": gamma,
        "memory": snapshot(),
    }


def run_speculative_suite(
    target_model_id: str,
    draft_model_id: str,
    *,
    gammas: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    measure_runs: int = 2,
    warmup_runs: int = 0,
    seed: int = 42,
    prompt: str = "The future of efficient language model inference is",
    target_model: Any = None,
    draft_model: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> List[ExperimentRecord]:
    env = collect_environment(seed=seed)
    device = select_device()
    set_seed(seed)
    gammas = gammas or [2, 4]
    results: List[ExperimentRecord] = []

    try:
        target = load_transformers(
            target_model_id,
            method="fp32" if device == "cpu" else "fp16",
            device=device,
            model=target_model,
            tokenizer=tokenizer,
            **kwargs,
        )
        draft = load_transformers(
            draft_model_id,
            method="fp32" if device == "cpu" else "fp16",
            device=device,
            model=draft_model,
            tokenizer=tokenizer or target._tokenizer,
            **kwargs,
        )
    except Exception as exc:
        return [
            unsupported(
                experiment_id=f"spec_{uuid.uuid4().hex[:8]}",
                experiment_type="speculative_decoding",
                model_id=target_model_id,
                backend="transformers",
                method="speculative",
                device=device,
                precision="fp32",
                reason=f"could not load target/draft: {type(exc).__name__}: {exc}",
                environment=env,
                config={"draft_model_id": draft_model_id, "gammas": gammas},
            )
        ]

    ok, why = _vocab_compatible(target._impl, draft._impl)
    if not ok:
        target.close()
        draft.close()
        return [
            unsupported(
                experiment_id=f"spec_{uuid.uuid4().hex[:8]}",
                experiment_type="speculative_decoding",
                model_id=target_model_id,
                backend="transformers",
                method="speculative",
                device=device,
                precision=target.precision,
                reason=why,
                environment=env,
                config={"draft_model_id": draft_model_id},
            )
        ]

    try:
        baseline_samples = []
        for _ in range(max(0, warmup_runs)):
            generate_transformers(target, prompt, max_new_tokens=max_new_tokens)
        for _ in range(max(1, measure_runs)):
            baseline_samples.append(
                generate_transformers(target, prompt, max_new_tokens=max_new_tokens)
            )
        baseline_tps = [
            s.tokens_per_sec for s in baseline_samples if s.tokens_per_sec
        ]
        baseline_record = ExperimentRecord(
            experiment_id=f"spec_baseline_{uuid.uuid4().hex[:8]}",
            experiment_type="speculative_decoding",
            status=Status.MEASURED,
            timestamp_utc=utc_now(),
            model_id=target.model_id,
            backend="transformers",
            method="baseline",
            device=target.device,
            precision=target.precision,
            config={"draft_model_id": draft_model_id, "max_new_tokens": max_new_tokens},
            environment=env,
            metrics=BenchmarkMetrics(
                load_time_s=target.load_time_s,
                ttft_ms=percentile_stats([s.ttft_ms for s in baseline_samples]),
                e2e_latency_ms=percentile_stats([s.e2e_ms for s in baseline_samples]),
                tokens_per_sec=percentile_stats(baseline_tps),
                extra={"role": "baseline"},
            ),
            samples=[
                {
                    "ttft_ms": s.ttft_ms,
                    "e2e_ms": s.e2e_ms,
                    "tokens_per_sec": s.tokens_per_sec,
                    "completion_tokens": s.completion_tokens,
                }
                for s in baseline_samples
            ],
            notes=["Autoregressive target-only greedy decode."],
        )
        results.append(baseline_record)
        base_tps_mean = (
            baseline_record.metrics.tokens_per_sec.mean
            if baseline_record.metrics and baseline_record.metrics.tokens_per_sec
            else None
        )

        for gamma in gammas:
            runs = []
            try:
                for _ in range(max(1, measure_runs)):
                    runs.append(
                        speculative_generate(
                            target, draft, prompt, gamma=gamma, max_new_tokens=max_new_tokens
                        )
                    )
            except Exception as exc:
                results.append(
                    errored(
                        experiment_id=f"spec_g{gamma}_{uuid.uuid4().hex[:8]}",
                        experiment_type="speculative_decoding",
                        model_id=target.model_id,
                        backend="transformers",
                        method=f"speculative_gamma_{gamma}",
                        device=device,
                        precision=target.precision,
                        reason=f"{type(exc).__name__}: {exc}",
                        environment=env,
                        config={"gamma": gamma, "draft_model_id": draft_model_id},
                    )
                )
                continue

            tps = [r["tokens_per_sec"] for r in runs if r["tokens_per_sec"]]
            acc = [r["acceptance_rate"] for r in runs if r["acceptance_rate"] is not None]
            speedup = None
            if base_tps_mean and tps:
                speedup = (sum(tps) / len(tps)) / base_tps_mean
            results.append(
                ExperimentRecord(
                    experiment_id=f"spec_g{gamma}_{uuid.uuid4().hex[:8]}",
                    experiment_type="speculative_decoding",
                    status=Status.MEASURED,
                    timestamp_utc=utc_now(),
                    model_id=target.model_id,
                    backend="transformers",
                    method=f"speculative_gamma_{gamma}",
                    device=target.device,
                    precision=target.precision,
                    config={
                        "gamma": gamma,
                        "draft_model_id": draft.model_id,
                        "max_new_tokens": max_new_tokens,
                    },
                    environment=env,
                    metrics=BenchmarkMetrics(
                        load_time_s=target.load_time_s + draft.load_time_s,
                        ttft_ms=percentile_stats(
                            [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
                        ),
                        e2e_latency_ms=percentile_stats([r["e2e_ms"] for r in runs]),
                        tokens_per_sec=percentile_stats(tps),
                        extra={
                            "acceptance_rate_mean": sum(acc) / len(acc) if acc else None,
                            "speedup_over_baseline": speedup,
                            "drafted_tokens_mean": sum(r["drafted_tokens"] for r in runs)
                            / len(runs),
                            "accepted_tokens_mean": sum(r["accepted_tokens"] for r in runs)
                            / len(runs),
                        },
                    ),
                    samples=runs,
                    notes=[
                        "Greedy speculative decoding; temperature=0 for reproducibility.",
                        "Acceptance rate is drafted-token matches against target argmax.",
                        "Speedup is wall-clock tokens/sec versus the measured baseline on this machine.",
                    ],
                )
            )
    finally:
        target.close()
        draft.close()
    return results
