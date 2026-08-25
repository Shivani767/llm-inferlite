from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from research.memory import gpu_memory_mb, rss_mb
from research.env import collect_environment


class GPUTelemetry(BaseModel):
    gpu_id: int
    name: str
    utilization_gpu: Optional[float] = None
    utilization_mem: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    timestamp: float
    status: str = "measured"
    reason: Optional[str] = None


class GPUTelemetryEngine:
    def collect_metrics(self) -> List[GPUTelemetry]:
        import time

        env = collect_environment()
        torch_info = env.get("torch") or {}
        gpu = torch_info.get("gpu")
        mem = gpu_memory_mb()
        ts = time.time()

        if gpu:
            return [
                GPUTelemetry(
                    gpu_id=int(gpu.get("index", 0)),
                    name=gpu.get("name", "cuda"),
                    memory_used_mb=mem.get("allocated_mb"),
                    memory_total_mb=gpu.get("total_memory_mb"),
                    timestamp=ts,
                    status="measured",
                    reason="torch.cuda memory snapshot; NVML utilization/power not attached",
                )
            ]

        # MPS/CPU: report host RSS only, never a fake NVIDIA GPU.
        return [
            GPUTelemetry(
                gpu_id=0,
                name=env.get("machine") or "cpu",
                memory_used_mb=rss_mb(),
                timestamp=ts,
                status="unsupported",
                reason="No CUDA GPU detected. Host RSS is reported instead of NVML telemetry.",
            )
        ]

    def get_energy_efficiency(self, throughput: float, power_w: float) -> Optional[float]:
        if not power_w:
            return None
        return throughput / power_w
