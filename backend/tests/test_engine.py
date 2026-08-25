from research.engine import run_benchmark
from research.schema import Status
from tests.tiny_lm import TinyTokenizer, make_tiny_lm


def test_engine_measures_tiny_model():
    model = make_tiny_lm()
    tok = TinyTokenizer()
    rec = run_benchmark(
        model_id="tiny-in-memory",
        method="fp32",
        prompt="hi",
        max_new_tokens=4,
        warmup_runs=0,
        measure_runs=2,
        model=model,
        tokenizer=tok,
    )
    assert rec.status == Status.MEASURED
    assert rec.metrics is not None
    assert rec.metrics.ttft_ms is not None
    assert rec.metrics.ttft_ms.n == 2
    assert rec.metrics.e2e_latency_ms is not None
    assert rec.metrics.tokens_per_sec is not None
    assert rec.metrics.load_time_s is not None
    assert rec.metrics.load_time_s >= 0
    assert rec.samples
    assert rec.environment.get("python")


def test_engine_labels_unimplemented_methods():
    rec = run_benchmark(
        model_id="anything",
        method="squeeze_llm",
        max_new_tokens=4,
        warmup_runs=0,
        measure_runs=1,
    )
    assert rec.status == Status.UNSUPPORTED
    assert rec.metrics is None
    assert rec.reason
