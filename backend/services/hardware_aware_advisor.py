from typing import Dict, Any, List, Optional

from pydantic import BaseModel

from research.capabilities import probe


class AdvisorRequest(BaseModel):
    gpu: str = "local"
    memory_gb: Optional[float] = None
    latency_sla_ms: Optional[float] = None
    model_name: str = "gpt2"
    throughput_target: Optional[float] = None


class AdvisorRecommendation(BaseModel):
    kind: str = "heuristic_policy"
    measured: bool = False
    runtime: str
    quantization: str
    batch_size: int
    rationale: str
    disclaimer: str
    try_next: List[str]
    local_capabilities: Dict[str, Any]


class HardwareAwareAdvisor:
    """
    Qualitative deployment policy. Does not invent TTFT/TPS/memory numbers.
    Run InferLite benchmarks on the target machine for measurements.
    """

    def recommend(self, request: AdvisorRequest) -> AdvisorRecommendation:
        caps = probe()
        mem = request.memory_gb
        try_next = ["fp32 or fp16 transformers baseline"]
        if caps["experiments"]["dynamic_int8"]["supported"]:
            try_next.append("dynamic_int8 on CPU")
        if caps["experiments"]["gguf"]["supported"]:
            try_next.append("GGUF via llama.cpp if a .gguf file is provided")
        if caps["experiments"]["int4_bnb"]["supported"]:
            try_next.append("bitsandbytes INT4/NF4 on CUDA")
        if caps["experiments"]["awq"]["supported"]:
            try_next.append("load a pre-quantized AWQ checkpoint")
        if caps["experiments"]["gptq"]["supported"]:
            try_next.append("load a pre-quantized GPTQ checkpoint")

        if mem is not None and mem < 10:
            runtime, quant, batch = "llama.cpp", "gguf_q4_k_m", 1
            rationale = "Heuristic: small VRAM budgets usually start with GGUF Q4_K_M if llama.cpp is installed."
        elif caps.get("cuda"):
            runtime, quant, batch = "transformers", "int4_bnb", 1
            rationale = "Heuristic: CUDA is present, so bitsandbytes INT4 is a reasonable first measurement."
        else:
            runtime, quant, batch = "transformers", "fp32", 1
            rationale = "Heuristic: CPU/MPS path — measure fp32 (and dynamic int8 on CPU) before claiming compression wins."

        return AdvisorRecommendation(
            runtime=runtime,
            quantization=quant,
            batch_size=batch,
            rationale=rationale,
            disclaimer=(
                "This is a planning heuristic, not a benchmark. InferLite does not emit expected_latency_ms "
                "or expected_throughput_tps without a measurement on the target hardware."
            ),
            try_next=try_next,
            local_capabilities={
                "device": caps["device"],
                "cuda": caps["cuda"],
                "mps": caps["mps"],
            },
        )
