from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from inference.kv_cache_research_suite import (
    KVCacheResearchSuite,
    KVCacheStrategy,
    KVCacheMetrics,
)
from inference.speculative_decoding_engine import (
    SpeculativeDecodingEngine,
    SpeculativeDecodingMetrics,
)

router = APIRouter()

class KVCacheRequest(BaseModel):
    model_name: str = "Llama-3-8B"
    strategies: Optional[List[str]] = None
    context_length: int = 4096

class SpeculativeRequest(BaseModel):
    target_model: str = "Llama-3-8B"
    draft_model: str = "TinyLlama-1.1B"
    num_speculative_tokens: int = 5

@router.post("/kv-cache/compare", response_model=List[KVCacheMetrics])
async def compare_kv_cache_strategies(request: KVCacheRequest):
    """
    Compare KV cache strategies (Dynamic, Prefix, Sliding Window, Paged Attention).
    """
    try:
        suite = KVCacheResearchSuite(request.model_name)
        strategies = None
        if request.strategies:
            strategies = [KVCacheStrategy(s) for s in request.strategies]
        results = suite.run_full_comparison(strategies, request.context_length)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kv-cache/report")
async def get_kv_cache_report(model_name: str = "Llama-3-8B"):
    """
    Generate a publishable-style report of KV cache study.
    """
    try:
        suite = KVCacheResearchSuite(model_name)
        results = suite.run_full_comparison()
        report = suite.generate_report(results)
        return {"report": report, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/speculative/evaluate", response_model=SpeculativeDecodingMetrics)
async def evaluate_speculative_decoding(request: SpeculativeRequest):
    """
    Evaluate speculative decoding (TinyLlama + Llama-3 vs Llama-3 alone).
    """
    try:
        engine = SpeculativeDecodingEngine(request.target_model, request.draft_model)
        results = engine.evaluate(request.num_speculative_tokens)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/speculative/report")
async def get_speculative_report(
    target_model: str = "Llama-3-8B",
    draft_model: str = "TinyLlama-1.1B",
):
    """
    Generate a publishable-style report of speculative decoding.
    """
    try:
        engine = SpeculativeDecodingEngine(target_model, draft_model)
        report = engine.generate_report()
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kv-cache/strategies")
async def list_kv_cache_strategies():
    """
    List all supported KV cache strategies.
    """
    return {
        "strategies": [s.value for s in KVCacheStrategy],
        "metrics": ["context_length", "memory_usage_gb", "latency_avg_ms", "cache_hit_rate", "throughput_tps"],
    }
