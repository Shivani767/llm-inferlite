from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.session import get_db
from inference.engine import (
    VLLMProvider, 
    ONNXRuntimeProvider, 
    LlamaCppProvider, 
    TransformersProvider,
    InferenceProvider
)
from benchmarking.research_runner import ResearchBenchmarkRunner
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from core.config import settings
import random

router = APIRouter()

class BenchmarkRequest(BaseModel):
    model_path: str
    runtimes: List[str] # ["vllm", "onnx", "llama.cpp", "transformers"]
    prompt: str = "What is the capital of France?"
    sampling_params: Optional[Dict[str, Any]] = {"max_tokens": 128, "temperature": 0.7}
    num_runs: int = 3

@router.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """
    Trigger a multi-runtime benchmark for a specific model.
    Measures latency, throughput, TTFT, and memory.
    """
    results = {}
    
    for runtime in request.runtimes:
        try:
            provider: Optional[InferenceProvider] = None
            if runtime == "vllm":
                provider = VLLMProvider(request.model_path)
            elif runtime == "onnx":
                provider = ONNXRuntimeProvider(request.model_path)
            elif runtime == "llama.cpp":
                provider = LlamaCppProvider(request.model_path)
            elif runtime == "transformers":
                provider = TransformersProvider(request.model_path)
            
            if provider:
                runner = ResearchBenchmarkRunner(provider)
                results[runtime] = runner.run_inference_benchmark(
                    request.prompt, 
                    request.sampling_params, 
                    request.num_runs
                )
            else:
                results[runtime] = {"error": f"Runtime {runtime} not supported"}
                
        except Exception as e:
            results[runtime] = {"error": str(e)}
            
    return results

@router.get("/runtimes")
async def list_available_runtimes():
    """List all inference runtimes supported by InferLite."""
    return [
        {"id": "vllm", "name": "vLLM", "description": "High-throughput serving with PagedAttention"},
        {"id": "onnx", "name": "ONNX Runtime", "description": "Cross-platform optimization and execution"},
        {"id": "llama.cpp", "name": "llama.cpp", "description": "Efficient C++ inference with quantization"},
        {"id": "transformers", "name": "Transformers", "description": "HuggingFace baseline runtime"}
    ]
