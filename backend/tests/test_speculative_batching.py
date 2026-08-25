from research.experiments.batching import run_batching_suite
from research.experiments.speculative import run_speculative_suite
from research.schema import Status
from tests.tiny_lm import TinyTokenizer, make_tiny_lm


def test_speculative_measures_acceptance_and_baseline():
    target = make_tiny_lm(n_layer=2, seed=0)
    draft = make_tiny_lm(n_layer=1, n_embd=32, n_head=2, seed=1)
    tok = TinyTokenizer()
    recs = run_speculative_suite(
        "tiny-target",
        "tiny-draft",
        gammas=[2],
        max_new_tokens=6,
        measure_runs=1,
        target_model=target,
        draft_model=draft,
        tokenizer=tok,
    )
    methods = {r.method: r for r in recs}
    assert "baseline" in methods
    assert methods["baseline"].status == Status.MEASURED
    spec = methods.get("speculative_gamma_2")
    assert spec is not None
    assert spec.status == Status.MEASURED
    acc = (spec.metrics.extra or {}).get("acceptance_rate_mean")
    assert acc is None or (0.0 <= acc <= 1.0)
    assert spec.metrics.tokens_per_sec is not None


def test_batching_returns_measured_or_error_never_fake_vllm():
    model = make_tiny_lm()
    tok = TinyTokenizer()
    recs = run_batching_suite(
        "tiny",
        num_requests=2,
        max_batch_size=2,
        max_new_tokens=3,
        model=model,
        tokenizer=tok,
    )
    assert recs
    for rec in recs:
        assert rec.status in {Status.MEASURED, Status.ERROR}
        if rec.status == Status.MEASURED:
            assert rec.metrics is not None
            assert rec.metrics.e2e_latency_ms is not None
