"""Typed records for measured, unsupported, and failed experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Status(str, Enum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class PercentileStats(BaseModel):
    n: int
    mean: float
    std: float
    min: float
    p50: float
    p95: float
    p99: float
    max: float
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None


class BenchmarkMetrics(BaseModel):
    load_time_s: Optional[float] = None
    ttft_ms: Optional[PercentileStats] = None
    e2e_latency_ms: Optional[PercentileStats] = None
    inter_token_latency_ms: Optional[PercentileStats] = None
    tokens_per_sec: Optional[PercentileStats] = None
    prompt_tokens: Optional[int] = None
    completion_tokens_mean: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    peak_gpu_allocated_mb: Optional[float] = None
    peak_gpu_reserved_mb: Optional[float] = None
    model_weight_mb: Optional[float] = None
    perplexity: Optional[float] = None
    energy_j: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    experiment_id: str
    experiment_type: str
    status: Status
    timestamp_utc: str
    model_id: str
    backend: str
    method: str
    device: str
    precision: str
    config: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
    metrics: Optional[BenchmarkMetrics] = None
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_metrics_unless_measured(self) -> "ExperimentRecord":
        if self.status != Status.MEASURED:
            if self.metrics is not None:
                raise ValueError("metrics may only be set when status is 'measured'")
            if self.samples:
                raise ValueError("samples may only be set when status is 'measured'")
        else:
            if self.metrics is None:
                raise ValueError("measured experiments must include metrics")
        return self


def unsupported(
    *,
    experiment_id: str,
    experiment_type: str,
    model_id: str,
    backend: str,
    method: str,
    device: str,
    precision: str,
    reason: str,
    environment: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
    timestamp_utc: Optional[str] = None,
) -> ExperimentRecord:
    from research.env import utc_now

    return ExperimentRecord(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        status=Status.UNSUPPORTED,
        timestamp_utc=timestamp_utc or utc_now(),
        model_id=model_id,
        backend=backend,
        method=method,
        device=device,
        precision=precision,
        config=config or {},
        environment=environment,
        reason=reason,
        notes=notes or [],
    )


def errored(
    *,
    experiment_id: str,
    experiment_type: str,
    model_id: str,
    backend: str,
    method: str,
    device: str,
    precision: str,
    reason: str,
    environment: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
    timestamp_utc: Optional[str] = None,
) -> ExperimentRecord:
    from research.env import utc_now

    return ExperimentRecord(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        status=Status.ERROR,
        timestamp_utc=timestamp_utc or utc_now(),
        model_id=model_id,
        backend=backend,
        method=method,
        device=device,
        precision=precision,
        config=config or {},
        environment=environment,
        reason=reason,
        notes=notes or [],
    )
