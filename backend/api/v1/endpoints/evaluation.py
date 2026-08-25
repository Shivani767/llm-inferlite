from fastapi import APIRouter
from evaluation.research_evaluator import ResearchEvaluator
from research.storage import ResultStore
from research.experiments.pareto import pareto_front
from typing import List

router = APIRouter()


@router.post("/run/{model_id}")
async def run_model_evaluation(
    model_id: int,
    benchmarks: List[str] = ["mmlu", "gsm8k", "perplexity"],
    version_tag: str = "base",
):
    evaluator = ResearchEvaluator(f"model_{model_id}", version_tag)
    results = []
    if "mmlu" in benchmarks:
        results.append(await evaluator.run_mmlu())
    if "gsm8k" in benchmarks:
        results.append(await evaluator.run_gsm8k())
    if "humaneval" in benchmarks:
        results.append(await evaluator.run_humaneval())
    if "perplexity" in benchmarks:
        results.append(await evaluator.run_perplexity())
    return results


@router.get("/leaderboard")
async def get_research_leaderboard():
    """Leaderboard from stored measured experiments. Empty if you have not run a suite."""
    recs = ResultStore().load_all()
    measured = [r for r in recs if r.status.value == "measured" and r.metrics]
    rows = []
    for r in measured:
        m = r.metrics
        rows.append(
            {
                "model": r.model_id,
                "variant": r.method,
                "status": r.status.value,
                "tps": m.tokens_per_sec.mean if m.tokens_per_sec else None,
                "p95_ms": m.e2e_latency_ms.p95 if m.e2e_latency_ms else None,
                "ttft_ms": m.ttft_ms.mean if m.ttft_ms else None,
                "memory_mb": m.peak_gpu_allocated_mb or m.peak_rss_mb,
                "perplexity": m.perplexity,
            }
        )
    return {
        "n_measured": len(rows),
        "rows": rows,
        "pareto": [{k: v for k, v in item.items() if k != "record"} for item in pareto_front(measured)],
        "note": "No fabricated Llama-3-8B numbers. Run a suite to populate this list.",
    }
