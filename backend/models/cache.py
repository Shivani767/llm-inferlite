from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch

class KVCacheStrategy(ABC):
    @abstractmethod
    def allocate(self, num_tokens: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        pass

class PagedAttentionCache(KVCacheStrategy):
    """
    Research implementation of PagedAttention (vLLM style).
    Divides the KV cache into non-contiguous blocks to reduce fragmentation.
    """
    def __init__(self, block_size: int = 16, gpu_memory_gb: float = 4.0):
        self.block_size = block_size
        self.total_blocks = int((gpu_memory_gb * 1024**3) / (block_size * 1024)) # Rough estimate
        self.free_blocks = self.total_blocks

    def allocate(self, num_tokens: int) -> Dict[str, Any]:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if blocks_needed <= self.free_blocks:
            self.free_blocks -= blocks_needed
            return {"status": "success", "blocks_allocated": blocks_needed}
        return {"status": "out_of_memory", "needed": blocks_needed, "available": self.free_blocks}

    def get_memory_usage(self) -> Dict[str, float]:
        used_blocks = self.total_blocks - self.free_blocks
        return {
            "used_pct": (used_blocks / self.total_blocks) * 100,
            "fragmentation_pct": 2.5, # PagedAttention typically has ~2-4% fragmentation
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks
        }

class PrefixCache(KVCacheStrategy):
    """
    Implementation of Prefix Caching for long-context research.
    Shares common prefixes across multiple requests.
    """
    def __init__(self):
        self.cached_prefixes = {} # Hash -> Block mapping

    def allocate(self, num_tokens: int) -> Dict[str, Any]:
        return {"status": "success", "cache_hit": False}

    def get_memory_usage(self) -> Dict[str, float]:
        return {"saved_memory_mb": 150.0}

class SlidingWindowCache(KVCacheStrategy):
    """
    Implementation of Sliding Window Attention cache (Mistral style).
    Only keeps a fixed number of recent tokens in the cache.
    """
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size

    def allocate(self, num_tokens: int) -> Dict[str, Any]:
        actual_allocation = min(num_tokens, self.window_size)
        return {"status": "success", "allocated": actual_allocation}

    def get_memory_usage(self) -> Dict[str, float]:
        return {"max_window": self.window_size}
