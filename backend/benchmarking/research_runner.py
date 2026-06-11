import time
import torch
import psutil
from typing import Dict, Any, List, Optional
import numpy as np
from models.inference import InferenceProvider

class ResearchBenchmarkRunner:
    """
    Advanced benchmarking engine for research-grade LLM performance analysis.
    """
    def __init__(self, provider: InferenceProvider):
        self.provider = provider

    def run_inference_benchmark(
        self, 
        prompt: str, 
        sampling_params: Dict[str, Any],
        num_runs: int = 5
    ) -> Dict[str, Any]:
        latencies = []
        ttfts = []
        tbts = [] # Time Between Tokens
        token_counts = []
        
        gpu_mem_start = self.provider.get_gpu_memory_usage()
        
        for _ in range(num_runs):
            # We use streaming to measure TTFT and TBT accurately
            start_time = time.time()
            tokens_times = []
            
            for chunk in self.provider.stream(prompt, sampling_params):
                tokens_times.append(time.time())
            
            end_time = time.time()
            
            if len(tokens_times) > 0:
                ttft = tokens_times[0] - start_time
                ttfts.append(ttft)
                
                if len(tokens_times) > 1:
                    # Calculate average TBT
                    diffs = np.diff(tokens_times)
                    tbts.extend(diffs.tolist())
            
            latencies.append(end_time - start_time)
            # This is a bit tricky since stream might return chunks of tokens
            # For research purposes, we'll assume the provider gives token-level or near-token-level streaming
            # In a real vLLM/TRT-LLM setup, we'd count actual tokens
            token_counts.append(len(tokens_times)) 

        gpu_mem_end = self.provider.get_gpu_memory_usage()
        
        return {
            "latency": {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "avg": np.mean(latencies)
            },
            "ttft": {
                "avg": np.mean(ttfts) if ttfts else 0,
                "p50": np.percentile(ttfts, 50) if ttfts else 0
            },
            "tbt": {
                "avg": np.mean(tbts) if tbts else 0,
                "p50": np.percentile(tbts, 50) if tbts else 0
            },
            "throughput": {
                "tokens_per_sec": sum(token_counts) / sum(latencies) if sum(latencies) > 0 else 0
            },
            "memory": {
                "gpu_allocated_gb_delta": gpu_mem_end["allocated_gb"] - gpu_mem_start["allocated_gb"],
                "gpu_reserved_gb_max": gpu_mem_end["max_allocated_gb"]
            }
        }

    def run_speculative_benchmark(
        self,
        target_provider: InferenceProvider,
        draft_provider: InferenceProvider,
        prompt: str,
        sampling_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        from models.inference import SpeculativeDecodingProvider
        
        # 1. Baseline (Target only)
        self.provider = target_provider
        baseline = self.run_inference_benchmark(prompt, sampling_params)
        
        # 2. Speculative
        spec_provider = SpeculativeDecodingProvider(target_provider, draft_provider)
        self.provider = spec_provider
        speculative = self.run_inference_benchmark(prompt, sampling_params)
        
        return {
            "baseline": baseline,
            "speculative": speculative,
            "speedup": baseline["latency"]["avg"] / speculative["latency"]["avg"],
            "acceptance_rate": speculative.get("speculative_metrics", {}).get("acceptance_rate", 0)
        }

    def run_comparison(
        self, 
        providers: Dict[str, InferenceProvider], 
        prompt: str, 
        sampling_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        comparison_results = {}
        for name, provider in providers.items():
            comparison_results[name] = self.run_inference_benchmark(prompt, sampling_params)
        return comparison_results
