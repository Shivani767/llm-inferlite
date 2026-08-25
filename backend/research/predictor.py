"""Ridge performance predictor. Fit only on measured rows. Ablate feature groups."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.experiments.pareto import _value
from research.schema import ExperimentRecord, Status

TARGETS = ("p95_latency_ms", "tokens_per_sec", "memory_mb")
FEATURE_GROUPS = {
    "hardware": ("cuda", "mps", "gpu_mem_mb"),
    "quantization": ("method",),
    "workload": ("context_tokens", "max_new_tokens", "batch_size"),
}


def _device_flags(rec: ExperimentRecord) -> Tuple[float, float, float]:
    env = rec.environment or {}
    torch_info = env.get("torch") or {}
    cuda = 1.0 if torch_info.get("cuda_available") or rec.device == "cuda" else 0.0
    mps = 1.0 if torch_info.get("mps_available") or rec.device == "mps" else 0.0
    gpu = torch_info.get("gpu") or {}
    mem = gpu.get("total_memory_mb")
    gpu_mem = float(mem) if mem is not None else 0.0
    return cuda, mps, gpu_mem


def _workload_feats(rec: ExperimentRecord) -> Tuple[float, float, float]:
    cfg = rec.config or {}
    ctx = cfg.get("context_tokens") or cfg.get("context_length") or cfg.get("prompt_tokens")
    if ctx is None and rec.metrics and rec.metrics.prompt_tokens:
        ctx = rec.metrics.prompt_tokens
    nnew = cfg.get("max_new_tokens") or 0
    batch = cfg.get("batch_size") or cfg.get("num_requests") or 1
    return float(ctx or 0), float(nnew or 0), float(batch or 1)


def _method_vocab(records: Sequence[ExperimentRecord]) -> List[str]:
    return sorted({r.method for r in records})


def featurize(
    records: Sequence[ExperimentRecord],
    *,
    methods: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    drop = set(drop or [])
    vocab = list(methods or _method_vocab(records))
    names: List[str] = []
    if "quantization" not in drop:
        names.extend([f"method={m}" for m in vocab])
    if "hardware" not in drop:
        names.extend(["cuda", "mps", "gpu_mem_mb"])
    if "workload" not in drop:
        names.extend(["log_context", "log_max_new", "log_batch"])

    rows = []
    for rec in records:
        feats: List[float] = []
        if "quantization" not in drop:
            feats.extend([1.0 if rec.method == m else 0.0 for m in vocab])
        if "hardware" not in drop:
            feats.extend(_device_flags(rec))
        if "workload" not in drop:
            ctx, nnew, batch = _workload_feats(rec)
            feats.extend(
                [
                    float(np.log1p(max(ctx, 0.0))),
                    float(np.log1p(max(nnew, 0.0))),
                    float(np.log1p(max(batch, 0.0))),
                ]
            )
        rows.append(feats)
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return X, names


def _y(records: Sequence[ExperimentRecord], target: str) -> np.ndarray:
    vals = []
    for rec in records:
        v = _value(rec, target)
        vals.append(np.nan if v is None else float(v))
    return np.asarray(vals, dtype=np.float64)


def _add_bias(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> Optional[np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(mask.sum()) < 2:
        return None
    Xb = _add_bias(X[mask])
    yy = y[mask]
    d = Xb.shape[1]
    reg = l2 * np.eye(d)
    reg[-1, -1] = 0.0
    try:
        w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ yy)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(Xb, yy, rcond=None)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _add_bias(X) @ w


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    yt, yp = y_true[mask], y_pred[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {"n": int(mask.sum()), "mae": mae, "rmse": rmse, "r2": r2}


def _measured(records: Iterable[ExperimentRecord]) -> List[ExperimentRecord]:
    return [r for r in records if r.status == Status.MEASURED and r.metrics is not None]


class PerformancePredictor:
    """Independent ridge models for latency, throughput, and memory."""

    def __init__(self, l2: float = 1e-2, drop: Optional[Sequence[str]] = None):
        self.l2 = l2
        self.drop = list(drop or [])
        self.methods: List[str] = []
        self.feature_names: List[str] = []
        self.weights: Dict[str, np.ndarray] = {}

    def fit(self, records: Sequence[ExperimentRecord]) -> "PerformancePredictor":
        rows = _measured(records)
        self.methods = _method_vocab(rows)
        X, names = featurize(rows, methods=self.methods, drop=self.drop)
        self.feature_names = names
        self.weights = {}
        for target in TARGETS:
            w = ridge_fit(X, _y(rows, target), l2=self.l2)
            if w is not None:
                self.weights[target] = w
        return self

    def predict_records(self, records: Sequence[ExperimentRecord]) -> np.ndarray:
        X, _ = featurize(records, methods=self.methods, drop=self.drop)
        cols = []
        for target in TARGETS:
            w = self.weights.get(target)
            if w is None or X.size == 0:
                cols.append(np.full(len(records), np.nan))
            else:
                cols.append(ridge_predict(X, w))
        return np.column_stack(cols)

    def leave_one_out(self, records: Sequence[ExperimentRecord]) -> Dict[str, Any]:
        rows = _measured(records)
        report: Dict[str, Any] = {"n": len(rows), "drop": self.drop, "targets": {}}
        if len(rows) < 3:
            report["reason"] = "need at least 3 measured rows for leave-one-out"
            return report
        preds = {t: np.full(len(rows), np.nan) for t in TARGETS}
        for i in range(len(rows)):
            train = rows[:i] + rows[i + 1 :]
            model = PerformancePredictor(l2=self.l2, drop=self.drop).fit(train)
            pred = model.predict_records([rows[i]])[0]
            for j, t in enumerate(TARGETS):
                preds[t][i] = pred[j]
        for t in TARGETS:
            report["targets"][t] = regression_metrics(_y(rows, t), preds[t])
        return report


def ablation_study(records: Sequence[ExperimentRecord]) -> Dict[str, Any]:
    """Fit full model and drop hardware / quantization / workload features."""
    rows = _measured(records)
    variants = {
        "full": [],
        "no_hardware": ["hardware"],
        "no_quantization": ["quantization"],
        "no_workload": ["workload"],
    }
    out = {"n_measured": len(rows), "variants": {}}
    for name, drop in variants.items():
        pred = PerformancePredictor(drop=drop)
        out["variants"][name] = pred.leave_one_out(rows)
    return out
