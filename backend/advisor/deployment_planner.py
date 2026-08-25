from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import math

class DeploymentRecommendation(BaseModel):
    runtime: str
    quantization: str
    gpu_type: str
    gpu_count: int
    expected_tps: float
    monthly_cost_usd: float
    cost_per_1m_tokens_usd: float
    latency_sla_met: bool

class CostAwareDeploymentPlanner:
    """
    Module 11: Strategic planner for economic and performance optimization.
    Calculates the TCO (Total Cost of Ownership) for LLM serving.
    """
    
    # Industry standard pricing (simplified for research)
    GPU_PRICING = {
        "A100-80GB": 3.67, # USD per hour (On-demand)
        "H100-80GB": 5.12,
        "L40S-48GB": 1.25,
        "RTX4090-24GB": 0.50 # Consumer/Spot
    }

    def plan_deployment(
        self, 
        users: int, 
        tokens_per_user_day: int, 
        latency_sla_ms: int,
        budget_usd: float
    ) -> List[DeploymentRecommendation]:
        """
        Recommends deployment strategies based on traffic and budget.
        """
        total_daily_tokens = users * tokens_per_user_day
        required_tps = total_daily_tokens / (24 * 3600)
        
        recommendations = []
        
        # Scenario 1: High Performance (H100 + vLLM)
        recommendations.append(self._calculate_scenario(
            "vLLM", "FP16", "H100-80GB", required_tps, latency_sla_ms
        ))
        
        # Scenario 2: Cost Optimized (L40S + AWQ-INT4)
        recommendations.append(self._calculate_scenario(
            "vLLM", "AWQ-INT4", "L40S-48GB", required_tps, latency_sla_ms
        ))
        
        # Scenario 3: Extreme Cost (RTX4090 + llama.cpp)
        recommendations.append(self._calculate_scenario(
            "llama.cpp", "GGUF-Q4_K_M", "RTX4090-24GB", required_tps, latency_sla_ms
        ))
        
        return recommendations

    def _calculate_scenario(
        self, runtime: str, quant: str, gpu: str, target_tps: float, sla: int
    ) -> DeploymentRecommendation:
        # Research-based performance heuristics
        base_tps_per_gpu = 150.0 if "H100" in gpu else 80.0
        if "INT4" in quant or "Q4" in quant:
            base_tps_per_gpu *= 2.5 # Throughput gain from quantization
            
        gpus_needed = max(1, math.ceil(target_tps / base_tps_per_gpu))
        hourly_cost = self.GPU_PRICING[gpu] * gpus_needed
        monthly_cost = hourly_cost * 24 * 30
        
        # Calculate cost per million tokens
        tokens_per_hour = base_tps_per_gpu * gpus_needed * 3600
        cost_per_1m = (hourly_cost / tokens_per_hour) * 1_000_000
        
        return DeploymentRecommendation(
            runtime=runtime,
            quantization=quant,
            gpu_type=gpu,
            gpu_count=gpus_needed,
            expected_tps=base_tps_per_gpu * gpus_needed,
            monthly_cost_usd=monthly_cost,
            cost_per_1m_tokens_usd=cost_per_1m,
            latency_sla_met=True if "H100" in gpu or "INT4" in quant else False
        )
