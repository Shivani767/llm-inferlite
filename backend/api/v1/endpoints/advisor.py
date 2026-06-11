from fastapi import APIRouter, Depends, Query
from services.auto_tune import AutoTuningEngine, TuningConfig
from advisor.ai_advisor import AIResearchAdvisor, OptimizationRecommendation
from services.hardware_aware_advisor import (
    HardwareAwareAdvisor,
    AdvisorRequest,
    AdvisorRecommendation,
    GPUModel,
)
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

class AutoTuneRequest(BaseModel):
    model_id: str
    target_hardware: str = "A100-80GB"
    objectives: List[str] = ["latency", "throughput"]
    constraints: Dict[str, Any] = {"max_memory_gb": 40, "min_accuracy_pct": 98.5}

@router.post("/hardware-aware/recommend", response_model=AdvisorRecommendation)
async def get_hardware_aware_recommendation(request: AdvisorRequest):
    """
    Get a hardware-aware optimization recommendation (runtime, quantization, batch size).
    """
    try:
        advisor = HardwareAwareAdvisor()
        recommendation = advisor.recommend(request)
        return recommendation
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend", response_model=OptimizationRecommendation)
async def get_ai_recommendation(
    model_name: str = Query(..., description="Name of the model"),
    hardware: str = Query("A100-80GB", description="Target hardware"),
    latency_sla: int = Query(200, description="Latency SLA in ms"),
    budget: float = Query(1000.0, description="Monthly budget")
):
    """
    Module 12: AI-driven optimization advisor.
    Returns the single best configuration for a given model and hardware profile.
    """
    advisor = AIResearchAdvisor(benchmark_history=[]) # In real app, pass history from DB
    return advisor.get_recommendation(model_name, hardware, latency_sla, budget)

@router.post("/search")
async def search_optimal_configs(request: AutoTuneRequest):
    """
    Runs a search across the runtime-quantization-batch space.
    Returns a Pareto-front of optimal deployment strategies.
    """
    engine = AutoTuningEngine(
        runtimes=["vllm", "ort", "llama.cpp"],
        quantizations=["fp16", "int8", "int4-gptq", "int4-awq"],
        batch_sizes=[1, 8, 16, 32]
    )
    
    # Simulated search results based on historical benchmark data
    simulated_results = [
        {"runtime": "vllm", "quantization": "int4-awq", "batch_size": 32, "latency_ms": 45, "throughput_tps": 1200, "memory_gb": 22},
        {"runtime": "ort", "quantization": "fp16", "batch_size": 1, "latency_ms": 12, "throughput_tps": 85, "memory_gb": 48},
        {"runtime": "llama.cpp", "quantization": "int4-gptq", "batch_size": 8, "latency_ms": 32, "throughput_tps": 450, "memory_gb": 18},
    ]
    
    pareto_front = engine.find_pareto_front(simulated_results)
    
    return {
        "model_id": request.model_id,
        "search_space_size": len(engine.generate_search_space()),
        "pareto_optimal_configs": pareto_front,
        "recommendation": engine.recommend_best(simulated_results, priority="balanced")
    }

@router.get("/hardware-profiles")
async def get_hardware_profiles():
    """Returns supported hardware profiles for auto-tuning."""
    return [
        {"id": "A100-80GB", "vram": 80, "arch": "ampere"},
        {"id": "H100-80GB", "vram": 80, "arch": "hopper"},
        {"id": "L40S-48GB", "vram": 48, "arch": "ada"},
        {"id": "RTX4090-24GB", "vram": 24, "arch": "ada"}
    ]

@router.get("/gpu-models")
async def list_gpu_models():
    """List all supported GPU models for the hardware-aware advisor."""
    return {"gpu_models": [g.value for g in GPUModel]}
