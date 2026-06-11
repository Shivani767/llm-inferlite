import asyncio
import random
import math
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

class ParallelismType(str, Enum):
    NONE = "none"
    TENSOR_PARALLELISM = "tensor_parallelism"
    PIPELINE_PARALLELISM = "pipeline_parallelism"
    HYBRID = "hybrid"

class DistributedSimulationConfig(BaseModel):
    rps: float = 10.0
    duration_seconds: int = 60
    burstiness: float = 0.2
    concurrency_limit: int = 32
    num_gpus: int = 4
    parallelism_type: ParallelismType = ParallelismType.TENSOR_PARALLELISM
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    model_size_billion: float = 70.0
    base_latency_ms: float = 300.0
    tokens_per_request: int = 128

class DistributedSimulationResult(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    max_concurrency: int
    queue_wait_avg_ms: float
    gpu_utilization_pct: List[float]
    communication_overhead_pct: float
    memory_per_gpu_gb: float
    parallelism_efficiency: float

class SimulationConfig(BaseModel):
    rps: float = 10.0  # Requests per second
    duration_seconds: int = 60
    burstiness: float = 0.2  # 0 to 1, how much traffic deviates from Poisson
    concurrency_limit: int = 32
    nodes: int = 1
    latency_p50_ms: float = 200.0
    latency_p99_ms: float = 500.0
    tokens_per_request: int = 128

class SimulationResult(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    max_concurrency: int
    queue_wait_avg_ms: float

class DistributedInferenceSimulator:
    def __init__(self, config: DistributedSimulationConfig):
        self.config = config
        self.results: List[float] = []
        self.queue_waits: List[float] = []
        self.active_requests = 0
        self.max_seen_concurrency = 0
        self.gpu_activity = [0.0 for _ in range(config.num_gpus)]
        
    def _calculate_latency(self) -> float:
        """
        Calculate inference latency based on parallelism strategy.
        Models communication overhead and parallelization efficiency.
        """
        base = self.config.base_latency_ms
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        
        # Tensor Parallelism: reduces compute but adds communication
        if self.config.parallelism_type in [ParallelismType.TENSOR_PARALLELISM, ParallelismType.HYBRID]:
            compute_reduction = 1.0 / tp_size
            comm_overhead = 0.05 * tp_size  # ~5% overhead per TP degree
            base *= (compute_reduction + comm_overhead)
            
        # Pipeline Parallelism: uses micro-batches, adds bubble overhead
        if self.config.parallelism_type in [ParallelismType.PIPELINE_PARALLELISM, ParallelismType.HYBRID]:
            bubble_factor = (pp_size - 1) / pp_size  # Pipeline bubble
            base *= (1 + bubble_factor * 0.3)  # 30% of bubble time adds to latency
            
        # Add token-dependent processing time
        token_factor = self.config.tokens_per_request / 128.0
        base *= token_factor
        
        # Add some variance
        variance = random.gauss(1.0, 0.1)
        return max(10, base * variance)
    
    def _calculate_memory_per_gpu(self) -> float:
        """Calculate memory required per GPU based on model size and parallelism."""
        model_memory = self.config.model_size_billion * 2.0  # ~2GB per B params for FP16
        kv_cache_memory = self.config.tokens_per_request * self.config.concurrency_limit * 2 * 4 / (1024 ** 3)  # ~GB
        
        if self.config.parallelism_type == ParallelismType.TENSOR_PARALLELISM:
            return (model_memory / self.config.tensor_parallel_size) + kv_cache_memory
        elif self.config.parallelism_type == ParallelismType.PIPELINE_PARALLELISM:
            return (model_memory / self.config.pipeline_parallel_size) + kv_cache_memory
        elif self.config.parallelism_type == ParallelismType.HYBRID:
            total_parallel = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
            return (model_memory / total_parallel) + kv_cache_memory
        else:
            return model_memory + kv_cache_memory
    
    def _calculate_communication_overhead(self) -> float:
        """Calculate percentage of time spent on communication."""
        if self.config.parallelism_type == ParallelismType.NONE:
            return 0.0
        elif self.config.parallelism_type == ParallelismType.TENSOR_PARALLELISM:
            return 5.0 * self.config.tensor_parallel_size
        elif self.config.parallelism_type == ParallelismType.PIPELINE_PARALLELISM:
            return 3.0 * self.config.pipeline_parallel_size
        else:  # HYBRID
            return (5.0 * self.config.tensor_parallel_size) + (3.0 * self.config.pipeline_parallel_size)
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel efficiency (1.0 = perfect scaling)."""
        if self.config.parallelism_type == ParallelismType.NONE:
            return 1.0
        
        tp_eff = 1.0 - (0.05 * (self.config.tensor_parallel_size - 1))
        pp_eff = 1.0 - (0.08 * (self.config.pipeline_parallel_size - 1))
        
        if self.config.parallelism_type == ParallelismType.TENSOR_PARALLELISM:
            return max(0.5, tp_eff)
        elif self.config.parallelism_type == ParallelismType.PIPELINE_PARALLELISM:
            return max(0.5, pp_eff)
        else:
            return max(0.4, tp_eff * pp_eff)
    
    async def run_simulation(self) -> DistributedSimulationResult:
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        
        tasks = []
        total_spawned = 0
        
        while time.time() < end_time:
            u = random.random()
            inter_arrival = -math.log(u) / self.config.rps
            if self.config.burstiness > 0:
                inter_arrival *= (1 + (random.random() - 0.5) * self.config.burstiness * 2)
            await asyncio.sleep(inter_arrival)
            tasks.append(asyncio.create_task(self._process_request()))
            total_spawned += 1
            if len(tasks) > 1000:
                tasks = [t for t in tasks if not t.done()]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Calculate GPU utilization
        gpu_util = [min(100.0, random.uniform(60, 95)) for _ in range(self.config.num_gpus)]
        
        if not self.results:
            return DistributedSimulationResult(
                total_requests=total_spawned,
                successful_requests=0,
                failed_requests=total_spawned,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_rps=0,
                max_concurrency=0,
                queue_wait_avg_ms=0,
                gpu_utilization_pct=gpu_util,
                communication_overhead_pct=self._calculate_communication_overhead(),
                memory_per_gpu_gb=self._calculate_memory_per_gpu(),
                parallelism_efficiency=self._calculate_parallel_efficiency()
            )
        
        self.results.sort()
        n = len(self.results)
        
        return DistributedSimulationResult(
            total_requests=total_spawned,
            successful_requests=n,
            failed_requests=total_spawned - n,
            avg_latency_ms=sum(self.results) / n,
            p50_latency_ms=self.results[int(n * 0.5)],
            p95_latency_ms=self.results[int(n * 0.95)],
            p99_latency_ms=self.results[int(n * 0.99)],
            throughput_rps=n / duration,
            max_concurrency=self.max_seen_concurrency,
            queue_wait_avg_ms=sum(self.queue_waits) / len(self.queue_waits) if self.queue_waits else 0,
            gpu_utilization_pct=gpu_util,
                communication_overhead_pct=self._calculate_communication_overhead(),
                memory_per_gpu_gb=self._calculate_memory_per_gpu(),
                parallelism_efficiency=self._calculate_parallel_efficiency()
        )
    
    async def _process_request(self):
        request_start = time.time()
        wait_start = time.time()
        max_conc = self.config.concurrency_limit * (self.config.num_gpus // max(self.config.tensor_parallel_size, self.config.pipeline_parallel_size, 1))
        
        while self.active_requests >= max_conc:
            await asyncio.sleep(0.01)
            if time.time() - wait_start > 5.0:
                return
                
        wait_duration = (time.time() - wait_start) * 1000
        self.queue_waits.append(wait_duration)
        self.active_requests += 1
        self.max_seen_concurrency = max(self.max_seen_concurrency, self.active_requests)
        
        try:
            processing_time_ms = self._calculate_latency()
            await asyncio.sleep(processing_time_ms / 1000.0)
            total_latency = (time.time() - request_start) * 1000
            self.results.append(total_latency)
        finally:
            self.active_requests -= 1

class ServingSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: List[float] = []
        self.queue_waits: List[float] = []
        self.active_requests = 0
        self.max_seen_concurrency = 0

    async def run_simulation(self) -> SimulationResult:
        """
        Simulates traffic and processing logic.
        Uses a discrete event simulation approach (simplified for async).
        """
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        
        tasks = []
        total_spawned = 0
        
        while time.time() < end_time:
            # Poisson arrival: inter-arrival time = -ln(U) / lambda
            # where U is uniform(0,1)
            u = random.random()
            inter_arrival = -math.log(u) / self.config.rps
            
            # Apply burstiness (adjusting inter-arrival)
            if self.config.burstiness > 0:
                inter_arrival *= (1 + (random.random() - 0.5) * self.config.burstiness * 2)
            
            await asyncio.sleep(inter_arrival)
            
            tasks.append(asyncio.create_task(self._process_request()))
            total_spawned += 1
            
            # Cleanup finished tasks periodically
            if len(tasks) > 1000:
                tasks = [t for t in tasks if not t.done()]

        # Wait for all simulated requests to finish (with a timeout)
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        if not self.results:
            return SimulationResult(
                total_requests=total_spawned,
                successful_requests=0,
                failed_requests=total_spawned,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_rps=0,
                max_concurrency=0,
                queue_wait_avg_ms=0
            )

        self.results.sort()
        n = len(self.results)
        
        return SimulationResult(
            total_requests=total_spawned,
            successful_requests=n,
            failed_requests=total_spawned - n,
            avg_latency_ms=sum(self.results) / n,
            p50_latency_ms=self.results[int(n * 0.5)],
            p95_latency_ms=self.results[int(n * 0.95)],
            p99_latency_ms=self.results[int(n * 0.99)],
            throughput_rps=n / duration,
            max_concurrency=self.max_seen_concurrency,
            queue_wait_avg_ms=sum(self.queue_waits) / len(self.queue_waits) if self.queue_waits else 0
        )

    async def _process_request(self):
        request_start = time.time()
        
        # Queueing logic
        wait_start = time.time()
        while self.active_requests >= (self.config.concurrency_limit * self.config.nodes):
            await asyncio.sleep(0.01) # Poll queue
            if time.time() - wait_start > 5.0: # 5s timeout
                return # Failed request (timeout)
        
        wait_duration = (time.time() - wait_start) * 1000
        self.queue_waits.append(wait_duration)
        
        self.active_requests += 1
        self.max_seen_concurrency = max(self.max_seen_concurrency, self.active_requests)
        
        try:
            # Simulate processing time
            # Lognormal distribution is often used for service times
            # Mean = p50, but we'll simplify with a normal distribution around p50
            mu = self.config.latency_p50_ms
            sigma = (self.config.latency_p99_ms - self.config.latency_p50_ms) / 2.33 # rough z-score for p99
            
            processing_time_ms = max(10, random.gauss(mu, sigma))
            await asyncio.sleep(processing_time_ms / 1000.0)
            
            total_latency = (time.time() - request_start) * 1000
            self.results.append(total_latency)
        finally:
            self.active_requests -= 1
