"""JSON + CSV persistence for experiment records. Never writes invented metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from research.schema import ExperimentRecord

RecordLike = Union[ExperimentRecord, Dict[str, Any]]


def _as_dict(record: RecordLike) -> Dict[str, Any]:
    if isinstance(record, ExperimentRecord):
        return record.model_dump()
    return dict(record)


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


class ResultStore:
    def __init__(self, root: Optional[Union[str, Path]] = None):
        self.root = Path(root) if root else default_results_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.root / "json"
        self.json_dir.mkdir(exist_ok=True)
        self.csv_path = self.root / "experiments.csv"

    def save(self, record: RecordLike) -> Path:
        data = _as_dict(record)
        exp_id = data.get("experiment_id") or "unknown"
        path = self.json_dir / f"{exp_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        self._append_csv(data)
        return path

    def save_many(self, records: Iterable[RecordLike]) -> List[Path]:
        return [self.save(r) for r in records]

    def load_all(self) -> List[ExperimentRecord]:
        records: List[ExperimentRecord] = []
        for path in sorted(self.json_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(ExperimentRecord.model_validate(data))
        return records

    def load_json_file(self, path: Union[str, Path]) -> List[ExperimentRecord]:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [ExperimentRecord.model_validate(x) for x in data]
        if isinstance(data, dict) and "records" in data:
            return [ExperimentRecord.model_validate(x) for x in data["records"]]
        return [ExperimentRecord.model_validate(data)]

    def write_bundle(self, records: Iterable[RecordLike], name: str = "suite") -> Path:
        payload = [_as_dict(r) for r in records]
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def _append_csv(self, data: Dict[str, Any]) -> None:
        metrics = data.get("metrics") or {}
        ttft = metrics.get("ttft_ms") or {}
        e2e = metrics.get("e2e_latency_ms") or {}
        tps = metrics.get("tokens_per_sec") or {}
        extra = metrics.get("extra") or {}
        row = {
            "experiment_id": data.get("experiment_id"),
            "timestamp_utc": data.get("timestamp_utc"),
            "experiment_type": data.get("experiment_type"),
            "status": data.get("status"),
            "reason": data.get("reason"),
            "model_id": data.get("model_id"),
            "backend": data.get("backend"),
            "method": data.get("method"),
            "device": data.get("device"),
            "precision": data.get("precision"),
            "load_time_s": metrics.get("load_time_s"),
            "ttft_ms_mean": ttft.get("mean"),
            "ttft_ms_p50": ttft.get("p50"),
            "ttft_ms_p95": ttft.get("p95"),
            "e2e_ms_mean": e2e.get("mean"),
            "e2e_ms_p50": e2e.get("p50"),
            "e2e_ms_p95": e2e.get("p95"),
            "e2e_ms_p99": e2e.get("p99"),
            "tokens_per_sec_mean": tps.get("mean"),
            "peak_rss_mb": metrics.get("peak_rss_mb"),
            "peak_gpu_allocated_mb": metrics.get("peak_gpu_allocated_mb"),
            "model_weight_mb": metrics.get("model_weight_mb"),
            "perplexity": metrics.get("perplexity"),
            "acceptance_rate_mean": extra.get("acceptance_rate_mean"),
            "speedup_over_baseline": extra.get("speedup_over_baseline"),
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
