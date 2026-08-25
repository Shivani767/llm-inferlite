from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from inference.kv_cache_research_suite import KVCacheResearchSuite, KVCacheStrategy
from inference.speculative_decoding_engine import SpeculativeDecodingEngine

router = APIRouter()


class KVCacheRequest(BaseModel):
    model_name: str = "gpt2"
    strategies: Optional[List[str]] = None
    context_length: int = 128


class SpeculativeRequest(BaseModel):
    target_model: str = "gpt2"
    draft_model: str = "distilgpt2"
    num_speculative_tokens: int = 4
    max_new_tokens: int = 32


@router.post("/kv-cache/compare")
async def compare_kv_cache_strategies(request: KVCacheRequest):
    try:
        suite = KVCacheResearchSuite(request.model_name)
        return suite.run_full_comparison(request.strategies, request.context_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kv-cache/report")
async def get_kv_cache_report(model_name: str = "gpt2"):
    try:
        suite = KVCacheResearchSuite(model_name)
        results = suite.run_full_comparison()
        return {"report": suite.generate_report(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speculative/evaluate")
async def evaluate_speculative_decoding(request: SpeculativeRequest):
    try:
        engine = SpeculativeDecodingEngine(request.target_model, request.draft_model)
        return engine.evaluate(request.num_speculative_tokens, max_new_tokens=request.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speculative/report")
async def get_speculative_report(
    target_model: str = "gpt2",
    draft_model: str = "distilgpt2",
):
    try:
        engine = SpeculativeDecodingEngine(target_model, draft_model)
        return {"report": engine.generate_report()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kv-cache/strategies")
async def list_kv_cache_strategies():
    return {
        "strategies": KVCacheStrategy.values(),
        "note": "paged_attention is measured only when vLLM is installed; otherwise it is labeled unsupported.",
    }
