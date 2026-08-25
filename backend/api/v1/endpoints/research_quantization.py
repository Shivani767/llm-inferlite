from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from quantization.quantization_research_suite import QuantizationResearchSuite
from quantization.research_engines import QuantizationResearchManager
from research.capabilities import probe
from research.experiments.quantization import DEFAULT_METHODS

router = APIRouter()


class QuantizationExperimentRequest(BaseModel):
    model_name: str = "gpt2"
    methods: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = {}


class OldQuantizationExperimentRequest(BaseModel):
    model_id: str
    method: str
    config: Optional[Dict[str, Any]] = {}
    run_eval: bool = False


@router.post("/run")
async def run_quantization_experiment(request: OldQuantizationExperimentRequest):
    manager = QuantizationResearchManager()
    try:
        result = manager.run_quantization_experiment(
            request.method, request.model_id, request.config or {}
        )
        return {"experiment_id": result.get("experiment_id"), "status": result.get("status"), "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suite/comparison")
async def run_quantization_comparison(request: QuantizationExperimentRequest):
    try:
        suite = QuantizationResearchSuite(request.model_name)
        return suite.run_full_comparison(request.methods)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suite/report")
async def get_quantization_report(model_name: str = "gpt2"):
    try:
        suite = QuantizationResearchSuite(model_name)
        results = suite.run_full_comparison()
        return {"report": suite.generate_publishable_report(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suite/pareto")
async def get_pareto_optimal(model_name: str = "gpt2"):
    try:
        suite = QuantizationResearchSuite(model_name)
        results = suite.run_full_comparison()
        return suite.get_pareto_optimal(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def list_supported_methods():
    caps = probe()
    return {
        "methods": DEFAULT_METHODS,
        "capability_matrix": caps["experiments"],
        "device": caps["device"],
        "note": "Methods that are not supported on this machine return status=unsupported and no scores.",
    }
