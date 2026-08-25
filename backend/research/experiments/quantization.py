"""Quantization experiments: measure when possible, otherwise label unsupported."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from research.capabilities import probe
from research.engine import run_benchmark
from research.env import collect_environment, select_device
from research.quality import compute_perplexity
from research.schema import ExperimentRecord, Status

DEFAULT_METHODS = [
    "fp32",
    "fp16",
    "bf16",
    "dynamic_int8",
    "int8_bnb",
    "int4_bnb",
    "awq",
    "gptq",
    "gguf_q4_k_m",
    "smooth_quant",
    "squeeze_llm",
]


def run_quantization_suite(
    model_id: str,
    methods: Optional[List[str]] = None,
    *,
    prompt: str = "The future of efficient language model inference is",
    max_new_tokens: int = 32,
    warmup_runs: int = 1,
    measure_runs: int = 3,
    seed: int = 42,
    gguf_repo: Optional[str] = None,
    gguf_file: Optional[str] = None,
    gptq_model_id: Optional[str] = None,
    awq_model_id: Optional[str] = None,
    **kwargs: Any,
) -> List[ExperimentRecord]:
    methods = methods or DEFAULT_METHODS
    env = collect_environment(seed=seed)
    device = select_device()
    results: List[ExperimentRecord] = []

    for method in methods:
        load_kwargs: Dict[str, Any] = dict(kwargs)
        mid = model_id
        backend = "transformers"
        m = method.lower()
        if m in {"gguf", "gguf_q4_k_m"}:
            backend = "llama.cpp"
            mid = gguf_repo or model_id
            load_kwargs["gguf_file"] = gguf_file
            load_kwargs["filename"] = gguf_file
        elif m == "gptq" and gptq_model_id:
            load_kwargs["quantized_model_id"] = gptq_model_id
            mid = gptq_model_id
        elif m == "awq" and awq_model_id:
            load_kwargs["quantized_model_id"] = awq_model_id
            mid = awq_model_id

        record = run_benchmark(
            model_id=mid,
            method=m,
            backend=backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            seed=seed,
            device=device,
            experiment_type="quantization",
            environment=env,
            **load_kwargs,
        )
        results.append(record)
    return results


def attach_perplexity(record: ExperimentRecord, model, tokenizer, device: str) -> ExperimentRecord:
    if record.status != Status.MEASURED or record.metrics is None:
        return record
    ppl = compute_perplexity(model, tokenizer, device)
    if ppl is not None:
        record.metrics.perplexity = ppl
        record.notes.append(
            "Perplexity is NLL exp() on a built-in ~100-token English passage, not MMLU."
        )
    return record


def capability_notes() -> Dict[str, Any]:
    return probe()
