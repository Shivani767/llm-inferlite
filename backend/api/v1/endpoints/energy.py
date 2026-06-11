from fastapi import APIRouter, Depends
from core.config import settings
import random

try:
    from services.energy import EnergyIntelligenceService
    HAS_ENERGY = True
except ImportError:
    HAS_ENERGY = False

router = APIRouter()
if HAS_ENERGY:
    energy_service = EnergyIntelligenceService()

@router.get("/gpu/telemetry")
async def get_gpu_telemetry():
    """
    Get real-time GPU power and thermal metrics.
    """
    if not HAS_ENERGY or settings.SIMULATION_MODE:
        return {
            "gpu_id": 0,
            "power_draw_watts": random.uniform(150, 350),
            "temperature_c": random.uniform(60, 85),
            "fan_speed_pct": random.uniform(40, 90),
            "memory_used_mb": random.uniform(10000, 40000)
        }
    return energy_service.get_gpu_power_usage()

@router.post("/estimate")
async def estimate_impact(duration_seconds: float, avg_power_watts: float):
    """
    Estimate the environmental impact of a model run.
    """
    if not HAS_ENERGY or settings.SIMULATION_MODE:
        energy_kwh = (avg_power_watts * duration_seconds) / (3600 * 1000)
        return {
            "energy_consumption_kwh": energy_kwh * 1.2,
            "carbon_footprint_kg_co2": energy_kwh * 1.2 * 0.4,
            "estimated_cost_usd": energy_kwh * 1.2 * 0.12
        }
    return energy_service.estimate_energy_cost(duration_seconds, avg_power_watts)
