from fastapi import APIRouter, HTTPException
from inference.cache_lab import KVCacheLab
from inference.speculative import SpeculativeDecodingLab
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter()

class CacheExperimentRequest(BaseModel):
    model_id: str
    strategy: str # paged, prefix, sliding_window
    params: Optional[Dict[str, Any]] = {}

class SpeculativeRequest(BaseModel):
    target_model_id: str
    draft_model_id: str
    gamma: int = 5

@router.post("/cache/simulate")
async def simulate_cache_strategy(request: CacheExperimentRequest):
    """
    Research endpoint to simulate advanced KV Cache strategies.
    Measures theoretical VRAM savings and throughput gains.
    """
    lab = KVCacheLab(request.model_id)
    
    if request.strategy == "paged":
        return lab.simulate_paged_attention(**request.params)
    elif request.strategy == "prefix":
        return lab.simulate_prefix_caching(**request.params)
    elif request.strategy == "sliding_window":
        return lab.simulate_sliding_window(**request.params)
    else:
        raise HTTPException(status_code=400, detail="Unsupported cache strategy")

@router.post("/speculative/run")
async def run_speculative_decoding_research(request: SpeculativeRequest):
    """
    Research endpoint for Speculative Decoding analysis.
    Calculates speedup factors and acceptance rates for draft models.
    """
    lab = SpeculativeDecodingLab(request.target_model_id, request.draft_model_id)
    return lab.run_speculative_experiment(gamma=request.gamma)

@router.get("/speculative/comparison")
async def compare_decoding_strategies(target_model_id: str):
    """
    Compares standard decoding vs various speculative decoding setups.
    """
    lab = SpeculativeDecodingLab(target_model_id, "draft-placeholder")
    return lab.compare_strategies()
