import os
from abc import ABC, abstractmethod
from typing import Dict, Any

from research.engine import run_benchmark


class QuantizationProvider(ABC):
    @abstractmethod
    async def quantize(self, model_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        pass


class HFQuantizer(QuantizationProvider):
    async def quantize(self, model_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        method = config.get("type") or config.get("method") or "fp32"
        rec = run_benchmark(
            model_id=model_path,
            method=method,
            max_new_tokens=int(config.get("max_new_tokens", 16)),
            measure_runs=int(config.get("measure_runs", 1)),
            warmup_runs=0,
            quantized_model_id=config.get("quantized_model_id"),
        )
        payload = rec.model_dump()
        payload["output_path"] = output_path
        return payload


class GGUFQuantizer(QuantizationProvider):
    async def quantize(self, model_path: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        rec = run_benchmark(
            model_id=model_path,
            method="gguf",
            backend="llama.cpp",
            gguf_file=config.get("gguf_file"),
            filename=config.get("gguf_file"),
            model_path=config.get("model_path") or (model_path if os.path.exists(model_path) else None),
            max_new_tokens=int(config.get("max_new_tokens", 16)),
            measure_runs=1,
            warmup_runs=0,
        )
        payload = rec.model_dump()
        payload["output_path"] = output_path
        return payload


class QuantizationEngine:
    def __init__(self):
        self.providers = {"hf": HFQuantizer(), "gguf": GGUFQuantizer()}

    async def run_quantization(self, provider_name: str, model_path: str, output_path: str, config: Dict[str, Any]):
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        return await provider.quantize(model_path, output_path, config)
