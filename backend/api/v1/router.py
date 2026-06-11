from fastapi import APIRouter

from api.v1.endpoints import (
    advisor,
    cost,
    energy,
    evaluation,
    experiments,
    inference,
    lab,
    models,
    onnx,
    reports,
    research_decoding,
    research_inference,
    research_labs,
    research_quantization,
    simulator,
)


api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["model-registry"])
api_router.include_router(research_inference.router, prefix="/research/inference", tags=["research-inference"])
api_router.include_router(research_quantization.router, prefix="/research/quantize", tags=["research-quantization"])
api_router.include_router(research_decoding.router, prefix="/research/decoding", tags=["research-decoding"])
api_router.include_router(cost.router, prefix="/cost", tags=["cost & planning"])
api_router.include_router(advisor.router, prefix="/advisor", tags=["advisor"])
api_router.include_router(simulator.router, prefix="/simulator", tags=["simulator"])
api_router.include_router(energy.router, prefix="/energy", tags=["energy intelligence"])
api_router.include_router(lab.router, prefix="/lab", tags=["onnx & observatory"])
api_router.include_router(reports.router, prefix="/reports", tags=["research reports"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference engine"])
api_router.include_router(onnx.router, prefix="/onnx", tags=["onnx pipeline"])
api_router.include_router(research_labs.router, prefix="/labs", tags=["research labs"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation framework"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiment tracking"])
