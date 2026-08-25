from typing import Any, Dict, Optional

from research.engine import run_benchmark


class QuantizationResearchManager:
    """Load/measure a quantization method or return unsupported. No canned scores."""

    def run_quantization_experiment(self, method: str, model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        rec = run_benchmark(
            model_id=model_id,
            method=method,
            backend="llama.cpp" if method.lower().startswith("gguf") else "transformers",
            max_new_tokens=int(config.get("max_new_tokens", 32)),
            measure_runs=int(config.get("measure_runs", 2)),
            warmup_runs=int(config.get("warmup_runs", 1)),
            gguf_file=config.get("gguf_file"),
            filename=config.get("gguf_file"),
            quantized_model_id=config.get("quantized_model_id"),
        )
        return rec.model_dump()
