"""Hardware-filtered inference search space. Unsupported methods are dropped, not scored."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence

from research.capabilities import is_supported


@dataclass(frozen=True)
class Candidate:
    method: str
    context_tokens: int
    max_new_tokens: int
    batch_size: int
    workload: str = "custom"

    @property
    def key(self) -> str:
        return (
            f"{self.method}|c{self.context_tokens}|n{self.max_new_tokens}|b{self.batch_size}"
        )

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        return data


METHOD_TO_CAP = {
    "fp32": "transformers_fp32",
    "fp16": "transformers_fp16",
    "bf16": "transformers_bf16",
    "dynamic_int8": "dynamic_int8",
    "int8_bnb": "int8_bnb",
    "int4_bnb": "int4_bnb",
    "awq": "awq",
    "gptq": "gptq",
    "gguf": "gguf",
    "gguf_q4_k_m": "gguf",
    "smooth_quant": "smoothquant",
    "squeeze_llm": "squeezellm",
}


def enumerate_space(
    *,
    methods: Sequence[str],
    context_tokens: Sequence[int],
    max_new_tokens: Sequence[int],
    batch_sizes: Optional[Sequence[int]] = None,
    workload: str = "custom",
    skip_unsupported: bool = True,
) -> List[Candidate]:
    batches = list(batch_sizes or [1])
    out: List[Candidate] = []
    skipped: List[Dict[str, str]] = []
    for method, ctx, nnew, bs in product(methods, context_tokens, max_new_tokens, batches):
        cap = METHOD_TO_CAP.get(method.lower())
        if skip_unsupported and cap:
            ok, reason = is_supported(cap)
            if not ok:
                skipped.append({"method": method, "reason": reason})
                continue
        out.append(
            Candidate(
                method=method,
                context_tokens=int(ctx),
                max_new_tokens=int(nnew),
                batch_size=int(bs),
                workload=workload,
            )
        )
    return out


def from_config(config: Dict[str, Any]) -> List[Candidate]:
    space = config.get("search_space") or {}
    methods = space.get("methods") or ["fp32", "fp16"]
    ctx = space.get("context_tokens") or space.get("context_lengths") or [32, 64]
    nnew = space.get("max_new_tokens") or [8, 16]
    if isinstance(nnew, int):
        nnew = [nnew]
    batches = space.get("batch_sizes") or [1]
    return enumerate_space(
        methods=methods,
        context_tokens=ctx,
        max_new_tokens=nnew,
        batch_sizes=batches,
        skip_unsupported=bool(space.get("skip_unsupported", True)),
    )


def keys(cands: Iterable[Candidate]) -> List[str]:
    return [c.key for c in cands]
