from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from research.experiments.kv_cache import run_kv_cache_suite
from research.experiments.speculative import run_speculative_suite

router = APIRouter()


class CacheExperimentRequest(BaseModel):
    model_id: str = "gpt2"
    strategy: str = "dynamic"
    context_length: int = 128
    params: Optional[Dict[str, Any]] = {}


class SpeculativeRequest(BaseModel):
    target_model_id: str = "gpt2"
    draft_model_id: str = "distilgpt2"
    gamma: int = 4
    max_new_tokens: int = 32


@router.post("/cache/simulate")
async def simulate_cache_strategy(request: CacheExperimentRequest):
    """
    Backward-compatible path. Runs a real KV-cache measurement for the named strategy.
    Historical name contains 'simulate'; results are measured or unsupported.
    """
    recs = run_kv_cache_suite(
        request.model_id,
        context_lengths=[request.context_length],
        strategies=[request.strategy],
        max_new_tokens=int((request.params or {}).get("max_new_tokens", 8)),
    )
    if not recs:
        raise HTTPException(status_code=400, detail="no results")
    return recs[0].model_dump()


@router.post("/speculative/run")
async def run_speculative_decoding_research(request: SpeculativeRequest):
    recs = run_speculative_suite(
        request.target_model_id,
        request.draft_model_id,
        gammas=[request.gamma],
        max_new_tokens=request.max_new_tokens,
        measure_runs=1,
    )
    spec = next((r for r in recs if r.method.startswith("speculative") or r.status.value != "measured"), recs[-1])
    return spec.model_dump()


@router.get("/speculative/comparison")
async def compare_decoding_strategies(target_model_id: str = "gpt2", draft_model_id: str = "distilgpt2"):
    recs = run_speculative_suite(target_model_id, draft_model_id, gammas=[4], max_new_tokens=16, measure_runs=1)
    return [r.model_dump() for r in recs]
