from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from research.capabilities import probe
from research.engine import run_benchmark
from research.env import collect_environment
from research.experiments.pareto import pareto_front
from research.runner import run_config
from research.storage import ResultStore
from research.viz import plot_suite

router = APIRouter()


class BenchRequest(BaseModel):
    model_id: str = "gpt2"
    method: str = "fp32"
    backend: str = "transformers"
    max_new_tokens: int = 32
    warmup_runs: int = 1
    measure_runs: int = 3
    prompt: Optional[str] = None
    gguf_file: Optional[str] = None


class SuiteRequest(BaseModel):
    config: Dict[str, Any]


@router.get("/capabilities")
async def research_capabilities():
    return probe()


@router.get("/environment")
async def research_environment():
    return collect_environment()


@router.post("/bench")
async def research_bench(request: BenchRequest):
    rec = run_benchmark(
        model_id=request.model_id,
        method=request.method,
        backend=request.backend,
        max_new_tokens=request.max_new_tokens,
        warmup_runs=request.warmup_runs,
        measure_runs=request.measure_runs,
        prompt=request.prompt or "The future of efficient language model inference is",
        gguf_file=request.gguf_file,
        filename=request.gguf_file,
    )
    ResultStore().save(rec)
    return rec.model_dump()


@router.post("/suite")
async def research_suite(request: SuiteRequest):
    try:
        summary = run_config(request.config, make_plots=True)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def list_results():
    recs = ResultStore().load_all()
    return [r.model_dump() for r in recs]


@router.post("/pareto")
async def research_pareto():
    recs = ResultStore().load_all()
    front = pareto_front(recs)
    return [{k: v for k, v in item.items() if k != "record"} for item in front]


@router.post("/plots")
async def research_plots():
    recs = ResultStore().load_all()
    paths = plot_suite(recs, ResultStore().root / "figures")
    return {"figures": [str(p) for p in paths]}
