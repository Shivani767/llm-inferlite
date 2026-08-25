from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from services.auto_tune import AutoTuningEngine
from advisor.ai_advisor import AIResearchAdvisor
from services.hardware_aware_advisor import (
    HardwareAwareAdvisor,
    AdvisorRequest,
    AdvisorRecommendation,
)
from research.storage import ResultStore
from research.experiments.pareto import pareto_front
from research.schema import ExperimentRecord, Status

router = APIRouter()


class AutoTuneRequest(BaseModel):
    model_id: str
    results: Optional[List[Dict[str, Any]]] = None
    results_dir: Optional[str] = None
    priority: str = "balanced"


@router.post("/hardware-aware/recommend", response_model=AdvisorRecommendation)
async def get_hardware_aware_recommendation(request: AdvisorRequest):
    advisor = HardwareAwareAdvisor()
    return advisor.recommend(request)


@router.get("/recommend")
async def get_ai_recommendation(
    model_name: str = Query(..., description="Name of the model"),
    hardware: str = Query("local", description="Target hardware"),
    latency_sla: int = Query(200, description="Latency SLA in ms"),
    budget: float = Query(1000.0, description="Monthly budget"),
):
    advisor = AIResearchAdvisor(benchmark_history=[])
    return advisor.get_recommendation(model_name, hardware, latency_sla, budget)


@router.post("/search")
async def search_optimal_configs(request: AutoTuneRequest):
    """Pareto search over *measured* records only. Does not synthesize scores."""
    records: List[ExperimentRecord] = []
    if request.results:
        records = [ExperimentRecord.model_validate(r) for r in request.results]
    elif request.results_dir:
        records = ResultStore(request.results_dir).load_all()
    else:
        records = ResultStore().load_all()

    measured = [r for r in records if r.status == Status.MEASURED]
    front = pareto_front(measured)
    engine = AutoTuningEngine(runtimes=[], quantizations=[], batch_sizes=[])
    as_dicts = []
    for rec in measured:
        m = rec.metrics
        if not m:
            continue
        as_dicts.append(
            {
                "runtime": rec.backend,
                "quantization": rec.method,
                "batch_size": rec.config.get("batch_size", 1),
                "latency_ms": m.e2e_latency_ms.p95 if m.e2e_latency_ms else None,
                "throughput_tps": m.tokens_per_sec.mean if m.tokens_per_sec else None,
                "memory_gb": (m.peak_gpu_allocated_mb or m.peak_rss_mb or 0) / 1024.0,
            }
        )
    as_dicts = [d for d in as_dicts if d["latency_ms"] is not None and d["throughput_tps"] is not None]
    rec_best = engine.recommend_best(as_dicts, priority=request.priority) if as_dicts else {}
    return {
        "model_id": request.model_id,
        "n_measured": len(measured),
        "n_excluded_unscored": len(records) - len(measured),
        "pareto_optimal_configs": [{k: v for k, v in item.items() if k != "record"} for item in front],
        "recommendation": rec_best,
        "note": "Empty Pareto means no measured results are stored yet. Run a suite first.",
    }


@router.get("/hardware-profiles")
async def get_hardware_profiles():
    return [
        {"id": "local", "note": "Use InferLite capabilities + benchmarks on the current machine"},
        {"id": "macbook", "note": "CPU or Apple MPS; GGUF if llama-cpp-python is installed"},
        {"id": "colab-t4", "note": "CUDA T4; bitsandbytes INT8/INT4 when installed"},
    ]


@router.get("/gpu-models")
async def list_gpu_models():
    return {"gpu_models": ["local", "macbook", "colab-t4", "unknown"]}
