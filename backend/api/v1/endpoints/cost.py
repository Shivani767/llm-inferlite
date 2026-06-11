from fastapi import APIRouter, Depends, Query
from telemetry.gpu_engine import GPUTelemetryEngine, GPUTelemetry
from advisor.deployment_planner import CostAwareDeploymentPlanner, DeploymentRecommendation
from typing import List

router = APIRouter()
gpu_engine = GPUTelemetryEngine()
planner = CostAwareDeploymentPlanner()

@router.get("/gpu/telemetry", response_model=List[GPUTelemetry])
async def get_realtime_telemetry():
    """
    Collect high-frequency hardware metrics (utilization, power, thermal).
    Essential for calculating tokens-per-joule and thermal throttling analysis.
    """
    return gpu_engine.collect_metrics()

@router.get("/gpu/efficiency")
async def get_efficiency_metrics(throughput: float = 100.0):
    """
    Calculate real-time energy efficiency of the current inference workload.
    Returns Tokens per Joule.
    """
    metrics = gpu_engine.collect_metrics()
    if not metrics:
        return {"error": "No GPU detected"}
    
    avg_power = sum(m.power_draw_w for m in metrics) / len(metrics)
    efficiency = gpu_engine.get_energy_efficiency(throughput, avg_power)
    
    return {
        "tokens_per_second": throughput,
        "avg_power_watts": avg_power,
        "tokens_per_joule": efficiency,
        "co2_grams_per_token": (avg_power / throughput) * 0.0001 # rough global avg
    }

@router.post("/planner/recommend", response_model=List[DeploymentRecommendation])
async def get_deployment_plan(
    users: int = Query(1000, description="Concurrent users"),
    tokens_per_user: int = Query(2000, description="Tokens per user per day"),
    latency_sla: int = Query(500, description="Target P99 latency in ms"),
    budget: float = Query(5000.0, description="Monthly budget in USD")
):
    """
    Cost-aware deployment planning engine.
    Optimizes for TCO while meeting performance SLAs.
    """
    return planner.plan_deployment(users, tokens_per_user, latency_sla, budget)
