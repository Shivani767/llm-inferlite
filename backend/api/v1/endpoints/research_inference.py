from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from benchmarking.runtime_benchmark_lab import (
    RuntimeBenchmarkLab,
    InferenceRuntime,
    RuntimeBenchmarkMetrics,
)

router = APIRouter()

class RuntimeBenchmarkRequest(BaseModel):
    model_name: str = "Llama-3-8B"
    runtimes: Optional[List[str]] = None  # Leave empty for all runtimes
    batch_size: int = 8
    concurrent_requests: int = 32
    prompt_length: int = 256
    max_new_tokens: int = 256

@router.post("/compare", response_model=List[RuntimeBenchmarkMetrics])
async def compare_runtimes(request: RuntimeBenchmarkRequest):
    """
    Run a full comparison of inference runtimes (vLLM, ONNX Runtime, llama.cpp, TensorRT-LLM).
    """
    try:
        lab = RuntimeBenchmarkLab(request.model_name)
        # Convert string runtimes to enum
        runtimes = None
        if request.runtimes:
            runtimes = [InferenceRuntime(r) for r in request.runtimes]
        results = lab.run_full_comparison(
            runtimes,
            request.batch_size,
            request.concurrent_requests,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report")
async def get_runtime_report(model_name: str = "Llama-3-8B"):
    """
    Generate a publishable-style report of runtime benchmark results.
    """
    try:
        lab = RuntimeBenchmarkLab(model_name)
        results = lab.run_full_comparison()
        report = lab.generate_report(results)
        return {"report": report, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runtimes")
async def list_supported_runtimes():
    """
    List all supported inference runtimes.
    """
    return {
        "runtimes": [r.value for r in InferenceRuntime],
        "metrics": ["ttft_ms", "tps", "latency_avg_ms", "latency_p95_ms", "latency_p99_ms", "memory_usage_gb"],
    }
