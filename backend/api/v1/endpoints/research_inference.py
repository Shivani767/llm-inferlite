from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from benchmarking.runtime_benchmark_lab import RuntimeBenchmarkLab, RUNTIME_TO_METHOD

router = APIRouter()


class RuntimeBenchmarkRequest(BaseModel):
    model_name: str = "gpt2"
    runtimes: Optional[List[str]] = None
    batch_size: int = 1
    concurrent_requests: int = 1
    prompt_length: int = 32
    max_new_tokens: int = 32
    gguf_file: Optional[str] = None


@router.post("/compare")
async def compare_runtimes(request: RuntimeBenchmarkRequest):
    try:
        lab = RuntimeBenchmarkLab(request.model_name)
        return lab.run_full_comparison(
            request.runtimes,
            max_new_tokens=request.max_new_tokens,
            gguf_file=request.gguf_file,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_runtime_report(model_name: str = "gpt2"):
    try:
        lab = RuntimeBenchmarkLab(model_name)
        results = lab.run_full_comparison()
        return {"report": lab.generate_report(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runtimes")
async def list_supported_runtimes():
    return {
        "runtimes": list(RUNTIME_TO_METHOD.keys()),
        "note": "Each runtime is probed. Missing engines return status=unsupported with a reason.",
        "metrics": ["ttft_ms", "tps", "latency_avg_ms", "latency_p95_ms", "latency_p99_ms", "memory_usage_gb"],
    }
