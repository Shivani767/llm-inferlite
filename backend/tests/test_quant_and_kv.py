from research.experiments.kv_cache import run_kv_cache_suite
from research.experiments.quantization import run_quantization_suite
from research.schema import Status
from tests.tiny_lm import TinyTokenizer, make_tiny_lm


def test_quantization_suite_does_not_fabricate_cuda_methods():
    recs = run_quantization_suite(
        "tiny",
        methods=["int8_bnb", "awq", "gptq", "smooth_quant", "squeeze_llm"],
        max_new_tokens=2,
        warmup_runs=0,
        measure_runs=1,
    )
    assert recs
    for rec in recs:
        assert rec.status in {Status.UNSUPPORTED, Status.ERROR, Status.MEASURED}
        if rec.status != Status.MEASURED:
            assert rec.metrics is None
            assert rec.reason
        # On CPU/Mac these CUDA methods must not appear as measured fake scores.
        if rec.method in {"smooth_quant", "squeeze_llm"}:
            assert rec.status == Status.UNSUPPORTED


def test_kv_cache_dynamic_is_measured_on_tiny_model():
    model = make_tiny_lm()
    tok = TinyTokenizer()
    recs = run_kv_cache_suite(
        "tiny-in-memory",
        context_lengths=[8, 16],
        max_new_tokens=2,
        measure_runs=1,
        strategies=["dynamic", "paged_attention"],
        model=model,
        tokenizer=tok,
    )
    statuses = {r.method: r.status for r in recs}
    assert Status.MEASURED in {r.status for r in recs if r.method == "dynamic"}
    assert statuses.get("paged_attention") in {Status.UNSUPPORTED, Status.MEASURED}
    paged = [r for r in recs if r.method == "paged_attention"]
    if paged and paged[0].status == Status.UNSUPPORTED:
        assert paged[0].metrics is None
        assert "vLLM" in (paged[0].reason or "")
