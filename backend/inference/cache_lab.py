"""KV cache lab: real measurements via the research suite; theoretical notes are labeled as such."""

from typing import Dict, Any, List

from research.experiments.kv_cache import run_kv_cache_suite


class KVCacheLab:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def simulate_paged_attention(self, block_size: int = 16, total_vram_gb: float = 24.0) -> Dict[str, Any]:
        recs = run_kv_cache_suite(self.model_id, strategies=["paged_attention"], context_lengths=[64], max_new_tokens=4)
        rec = recs[0].model_dump() if recs else {}
        rec["analytical_note"] = (
            "InferLite does not invent PagedAttention utilization. "
            "If vLLM is missing, status is unsupported."
        )
        rec["requested_block_size"] = block_size
        rec["requested_total_vram_gb"] = total_vram_gb
        return rec

    def simulate_prefix_caching(self, shared_prefix_tokens: int = 64) -> Dict[str, Any]:
        recs = run_kv_cache_suite(
            self.model_id,
            strategies=["prefix"],
            context_lengths=[shared_prefix_tokens],
            max_new_tokens=4,
        )
        return recs[0].model_dump() if recs else {"status": "error"}

    def simulate_sliding_window(self, window_size: int = 64) -> Dict[str, Any]:
        recs = run_kv_cache_suite(
            self.model_id,
            strategies=["sliding_window"],
            context_lengths=[window_size * 2],
            max_new_tokens=4,
        )
        return recs[0].model_dump() if recs else {"status": "error"}
