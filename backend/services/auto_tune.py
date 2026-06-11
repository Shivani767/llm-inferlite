from typing import List, Dict, Any, Optional
import itertools
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TuningConfig:
    runtime: str
    quantization: str
    batch_size: int
    kv_cache_strategy: str

class AutoTuningEngine:
    """
    Search engine for finding Pareto-optimal inference configurations.
    Optimizes for the Latency-Throughput-Cost frontier.
    """
    
    def __init__(self, runtimes: List[str], quantizations: List[str], batch_sizes: List[int]):
        self.runtimes = runtimes
        self.quantizations = quantizations
        self.batch_sizes = batch_sizes
        self.kv_strategies = ["dynamic", "paged", "prefix"]

    def generate_search_space(self) -> List[TuningConfig]:
        """Generates all valid combinations for exploration."""
        combinations = list(itertools.product(
            self.runtimes, 
            self.quantizations, 
            self.batch_sizes, 
            self.kv_strategies
        ))
        return [TuningConfig(*c) for c in combinations]

    def find_pareto_front(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify Pareto-optimal configurations from benchmark results.
        A config is Pareto-optimal if no other config is better in all objectives.
        """
        if not results:
            return []
            
        df = pd.DataFrame(results)
        
        # We want to minimize latency and memory, but maximize throughput
        # Convert throughput to negative for minimization
        df['neg_throughput'] = -df['throughput_tps']
        
        objectives = ['latency_ms', 'neg_throughput', 'memory_gb']
        pareto_mask = np.ones(df.shape[0], dtype=bool)
        
        for i, row in df.iterrows():
            # Find rows that strictly dominate the current row
            dominating = (
                (df[objectives] <= row[objectives]).all(axis=1) & 
                (df[objectives] < row[objectives]).any(axis=1)
            )
            if dominating.any():
                pareto_mask[i] = False
                
        pareto_front = df[pareto_mask].drop(columns=['neg_throughput']).to_dict(orient='records')
        return pareto_front

    def recommend_best(self, results: List[Dict[str, Any]], priority: str = "balanced") -> Dict[str, Any]:
        """Recommends the best configuration based on research priority."""
        pareto = self.find_pareto_front(results)
        if not pareto:
            return {}
            
        if priority == "latency":
            return min(pareto, key=lambda x: x['latency_ms'])
        elif priority == "throughput":
            return max(pareto, key=lambda x: x['throughput_tps'])
        elif priority == "memory":
            return min(pareto, key=lambda x: x['memory_gb'])
        else: # balanced: harmonic mean of normalized scores
            return pareto[0] # Placeholder for actual heuristic
