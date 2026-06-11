
import random
from typing import Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum
from quantization.quantization_research_suite import QuantizationMethod
from benchmarking.runtime_benchmark_lab import InferenceRuntime


class GPUModel(str, Enum):
    RTX_3090 = "rtx_3090"
    RTX_4090 = "rtx_4090"
    RTX_4060 = "rtx_4060"
    A10G = "a10g"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100 = "h100"


class AdvisorRequest(BaseModel):
    gpu: GPUModel
    memory_gb: Optional[float] = None  # If not provided, use GPU's default
    latency_sla_ms: float  # Required SLA for average latency
    model_name: str = "Llama-3-8B"
    throughput_target: Optional[float] = None


class AdvisorRecommendation(BaseModel):
    runtime: InferenceRuntime
    quantization: QuantizationMethod
    batch_size: int
    expected_latency_ms: float
    expected_throughput_tps: float
    expected_memory_gb: float
    meets_sla: bool
    rationale: str


class HardwareAwareAdvisor:
    """
    Hardware-aware optimization advisor.
    Takes GPU specs, latency SLA, and outputs optimal runtime/quantization/batch size.
    """

    # GPU specifications (memory in GB)
    GPU_SPECS = {
        GPUModel.RTX_3090: {"memory": 24, "compute_capability": "8.6", "speed_factor": 0.8},
        GPUModel.RTX_4090: {"memory": 24, "compute_capability": "8.9", "speed_factor": 1.0},
        GPUModel.RTX_4060: {"memory": 8, "compute_capability": "8.9", "speed_factor": 0.5},
        GPUModel.A10G: {"memory": 24, "compute_capability": "8.6", "speed_factor": 0.9},
        GPUModel.A100_40GB: {"memory": 40, "compute_capability": "8.0", "speed_factor": 1.3},
        GPUModel.A100_80GB: {"memory": 80, "compute_capability": "8.0", "speed_factor": 1.5},
        GPUModel.H100: {"memory": 80, "compute_capability": "9.0", "speed_factor": 2.0},
    }

    def __init__(self):
        pass

    def recommend(self, request: AdvisorRequest) -> AdvisorRecommendation:
        """
        Generate an optimal deployment recommendation.
        """
        gpu_specs = self.GPU_SPECS[request.gpu]
        available_memory = request.memory_gb or gpu_specs["memory"]
        
        # Decision logic
        if available_memory < 10:
            # Small GPU (<10GB): use llama.cpp + GGUF
            runtime = InferenceRuntime.LLAMA_CPP
            quantization = QuantizationMethod.GGUF_Q4_K_M
            batch_size = 4
            rationale = "Small GPU memory: using llama.cpp with GGUF Q4_K_M for minimal memory footprint"
        elif available_memory < 30:
            # Medium GPU (10-30GB): use vLLM + AWQ
            runtime = InferenceRuntime.VLLM
            quantization = QuantizationMethod.AWQ
            batch_size = 8
            rationale = "Medium GPU: using vLLM with AWQ for good balance of speed and memory"
        else:
            # Large GPU (>30GB): use TensorRT-LLM + optional quantization
            runtime = InferenceRuntime.TENSORRT_LLM
            quantization = QuantizationMethod.FP16
            batch_size = 16
            rationale = "Large GPU: using TensorRT-LLM for maximum throughput"
            
        # Calculate expected metrics (simplified)
        base_latency = 250 / gpu_specs["speed_factor"]
        if quantization in [QuantizationMethod.GPTQ, QuantizationMethod.AWQ, QuantizationMethod.GGUF_Q4_K_M]:
            latency_multiplier = 0.7
            memory_multiplier = 0.25
        elif quantization == QuantizationMethod.FP16:
            latency_multiplier = 1.0
            memory_multiplier = 1.0
        else:
            latency_multiplier = 0.85
            memory_multiplier = 0.5
            
        expected_latency = base_latency * latency_multiplier
        expected_throughput = (40 * gpu_specs["speed_factor"]) / latency_multiplier
        expected_memory = 16 * memory_multiplier
        
        meets_sla = expected_latency <= request.latency_sla_ms
        
        # Adjust batch size if needed for SLA
        if not meets_sla:
            batch_size = max(1, batch_size // 2)
            expected_latency *= 0.8
            meets_sla = expected_latency <= request.latency_sla_ms
            
        return AdvisorRecommendation(
            runtime=runtime,
            quantization=quantization,
            batch_size=batch_size,
            expected_latency_ms=expected_latency,
            expected_throughput_tps=expected_throughput,
            expected_memory_gb=expected_memory,
            meets_sla=meets_sla,
            rationale=rationale,
        )

