from research.experiments.pareto import hypervolume_throughput_memory, pareto_front
from research.optimizer import compare_strategies, hardware_heuristic, make_eval_fn, run_budget_sweep
from research.predictor import ablation_study
from research.schema import Status
from research.search_space import Candidate, enumerate_space
from tests.tiny_lm import TinyTokenizer, make_tiny_lm


def _space():
    return enumerate_space(
        methods=["fp32"],
        context_tokens=[8, 16],
        max_new_tokens=[2, 4],
        batch_sizes=[1],
        skip_unsupported=True,
    )


def test_search_space_drops_unsupported_cuda_methods():
    cands = enumerate_space(
        methods=["fp32", "int8_bnb", "squeeze_llm"],
        context_tokens=[8],
        max_new_tokens=[2],
        skip_unsupported=True,
    )
    methods = {c.method for c in cands}
    assert "fp32" in methods
    assert "squeeze_llm" not in methods


def test_optimizer_baselines_use_real_tiny_measurements():
    model = make_tiny_lm()
    tok = TinyTokenizer()
    cands = _space()
    assert len(cands) == 4
    evaluate = make_eval_fn(
        model_id="tiny-in-memory",
        seed=0,
        warmup_runs=0,
        measure_runs=1,
        tokenizer=tok,
        extra_load={"model": model, "tokenizer": tok},
    )
    study = compare_strategies(cands, evaluate, budget=3, seed=0)
    assert study["simulation"] is False
    grid = study["strategies"]["grid"]
    assert grid["n_evaluated"] == 4
    assert grid["n_measured"] >= 1
    for rec in grid["records"]:
        assert rec.status in {Status.MEASURED, Status.UNSUPPORTED, Status.ERROR}
        if rec.status != Status.MEASURED:
            assert rec.metrics is None
    rand = study["strategies"]["random"]
    assert rand["n_evaluated"] == 3
    heur = study["strategies"]["heuristic"]
    assert heur["n_evaluated"] == 1
    inf = study["strategies"]["inferlite"]
    assert inf["n_evaluated"] == 3
    assert inf["n_evaluated"] < grid["n_evaluated"]
    names = {row["strategy"] for row in study["comparison"]}
    assert names == {"grid", "random", "heuristic", "inferlite"}


def test_predictor_ablation_on_measured_rows():
    model = make_tiny_lm()
    tok = TinyTokenizer()
    cands = _space()
    evaluate = make_eval_fn(
        model_id="tiny-in-memory",
        seed=1,
        warmup_runs=0,
        measure_runs=1,
        tokenizer=tok,
        extra_load={"model": model, "tokenizer": tok},
    )
    recs = [evaluate(c) for c in cands]
    measured = [r for r in recs if r.status == Status.MEASURED]
    assert len(measured) >= 3
    report = ablation_study(measured)
    assert report["n_measured"] == len(measured)
    for name in ("full", "no_hardware", "no_quantization", "no_workload"):
        assert name in report["variants"]
        payload = report["variants"][name]
        assert "targets" in payload
    hv = hypervolume_throughput_memory(measured)
    assert hv is None or hv >= 0
    front = pareto_front(measured)
    assert front


def test_compare_strategies_loads_each_candidate_once():
    """Grid/random/heuristic/InferLite share measurements so Colab does not reload TinyLlama."""
    model = make_tiny_lm()
    tok = TinyTokenizer()
    cands = _space()
    inner = make_eval_fn(
        model_id="tiny-in-memory",
        seed=0,
        warmup_runs=0,
        measure_runs=1,
        tokenizer=tok,
        extra_load={"model": model, "tokenizer": tok},
    )
    keys: list[str] = []

    def counting(cand):
        keys.append(cand.key)
        return inner(cand)

    compare_strategies(cands, counting, budget=3, seed=0)
    assert keys
    assert len(keys) == len(set(keys))


def test_budget_sweep_replays_cached_measurements():
    """Grid is timed once; seed×budget strategies reuse those wall-clock records."""
    model = make_tiny_lm()
    tok = TinyTokenizer()
    cands = _space()
    inner = make_eval_fn(
        model_id="tiny-in-memory",
        seed=0,
        warmup_runs=0,
        measure_runs=1,
        tokenizer=tok,
        extra_load={"model": model, "tokenizer": tok},
    )
    keys: list[str] = []

    def counting(cand):
        keys.append(cand.key)
        return inner(cand)

    study = run_budget_sweep(
        cands,
        counting,
        seeds=[0, 1],
        budgets=[2, 4],
        strategies=["grid", "random", "heuristic", "inferlite"],
    )
    assert study["simulation"] is False
    assert study["replay"] is True
    assert len(keys) == len(cands)
    assert len(keys) == len(set(keys))
    assert study["n_search_space"] == 4
    assert study["budgets"] == [2, 4]
    assert len(study["sweep"]) == 2
    for row in study["sweep"]:
        inf = row["strategies"]["inferlite"]
        rnd = row["strategies"]["random"]
        assert inf["hypervolume"]["n"] == 2
        assert rnd["hypervolume"]["n"] == 2
        assert len(inf["per_seed"]) == 2
        assert inf["n_evaluated"] == row["budget"]
        assert rnd["n_evaluated"] == row["budget"]
        assert row["strategies"]["heuristic"]["n_evaluated"] == 1
    names = {row["strategy"] for row in study["comparison"]}
    assert names == {"grid", "random", "heuristic", "inferlite"}
    vs = {item["budget"]: item["inferlite_beats_random_on_mean"] for item in study["inferlite_vs_random"]}
    assert set(vs) == {2, 4}


def test_heuristic_prefers_supported_method():
    cands = [
        Candidate("fp32", 32, 8, 1),
        Candidate("fp32", 64, 8, 1),
    ]
    pick = hardware_heuristic(cands)
    assert pick.method == "fp32"
