"""Named workloads. Prompts are grown to a token budget; never invented metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BASE = "The future of efficient language model inference is "

WORKLOADS: Dict[str, Dict[str, Any]] = {
    "short": {
        "description": "short context, few new tokens, batch 1",
        "context_tokens": 32,
        "max_new_tokens": 8,
        "batch_size": 1,
    },
    "long": {
        "description": "longer prompt, still local-hardware sized",
        "context_tokens": 128,
        "max_new_tokens": 16,
        "batch_size": 1,
    },
    "low_concurrency": {
        "description": "short context, batch 1",
        "context_tokens": 48,
        "max_new_tokens": 8,
        "batch_size": 1,
    },
    "high_concurrency": {
        "description": "short context, static batch of 4 (if the engine path supports it)",
        "context_tokens": 48,
        "max_new_tokens": 8,
        "batch_size": 4,
    },
}


def list_workloads() -> List[str]:
    return list(WORKLOADS)


def prompt_for_tokens(
    n_tokens: int,
    *,
    tokenizer: Any = None,
    base: str = BASE,
) -> str:
    """Build a prompt with at least n_tokens according to tokenizer, else ~chars."""
    n_tokens = max(1, int(n_tokens))
    text = base
    if tokenizer is None:
        while len(text.split()) < n_tokens:
            text += base
        return text

    def _len(s: str) -> int:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(s)
            return len(ids) if not hasattr(ids, "shape") else int(ids.shape[-1])
        out = tokenizer(s)
        ids = out.get("input_ids") if isinstance(out, dict) else out
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        return len(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else len(ids)

    guard = 0
    while _len(text) < n_tokens and guard < 10_000:
        text += base
        guard += 1
    return text


def resolve_workload(
    name: Optional[str] = None,
    *,
    context_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    spec = dict(WORKLOADS.get(name or "short") or WORKLOADS["short"])
    if context_tokens is not None:
        spec["context_tokens"] = int(context_tokens)
    if max_new_tokens is not None:
        spec["max_new_tokens"] = int(max_new_tokens)
    if batch_size is not None:
        spec["batch_size"] = int(batch_size)
    spec["name"] = name or "custom"
    return spec
