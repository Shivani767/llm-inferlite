from fastapi import APIRouter
from core.config import settings

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
    """Real CUDA memory snapshot when available; otherwise status=unsupported."""
    if not HAS_ENERGY:
        return {"status": "unsupported", "reason": "energy service unavailable"}
    return energy_service.get_gpu_power_usage()


@router.post("/estimate")
async def estimate_impact(duration_seconds: float, avg_power_watts: float):
    """Carbon/cost estimate from caller-supplied watts. Does not invent power draw."""
    if not HAS_ENERGY:
        return {"status": "unsupported", "reason": "energy service unavailable"}
    return energy_service.estimate_energy_cost(duration_seconds, avg_power_watts)
