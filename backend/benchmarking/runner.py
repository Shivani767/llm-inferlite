import time
import torch
import psutil
from typing import Dict, Any, List
from models.inference import InferenceProvider

class BenchmarkRunner:
    def __init__(self, provider: InferenceProvider):
        self.provider = provider

    def run_latency_benchmark(self, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        # Measure TTFT (Time To First Token)
        # This is a simplified version
        first_token_time = None
        full_response = ""
        
        token_count = 0
        for token in self.provider.stream(prompt, config):
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += token
            token_count += 1
            
        total_time = time.time() - start_time
        
        return {
            "total_latency": total_time,
            "ttft": first_token_time,
            "tokens_per_second": token_count / total_time if total_time > 0 else 0,
            "total_tokens": token_count
        }

    def get_resource_usage(self) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "utilization": 0 # Would need pynvml for real utilization
            }
            
        return {
            "cpu_utilization": cpu,
            "memory_usage_percent": memory.percent,
            "gpu_info": gpu_info
        }

    def run_full_benchmark(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for scenario in scenarios:
            latency_metrics = self.run_latency_benchmark(
                scenario["prompt"], 
                scenario.get("config", {})
            )
            resource_metrics = self.get_resource_usage()
            
            results.append({
                "scenario": scenario["name"],
                **latency_metrics,
                **resource_metrics
            })
            
        return {
            "results": results,
            "summary": self._calculate_summary(results)
        }

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Calculate p50, p95, p99
        import numpy as np
        latencies = [r["total_latency"] for r in results]
        
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "avg_tokens_per_sec": sum(r["tokens_per_second"] for r in results) / len(results)
        }
