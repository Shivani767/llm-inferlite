from fastapi import APIRouter, HTTPException
from core.config import settings

try:
    from services.onnx_lab import ONNXCompilerLab
    from services.observatory import ModelInternalsObservatory
    HAS_LAB = True
except ImportError:
    HAS_LAB = False

router = APIRouter()

@router.get("/onnx/optimizations")
async def list_onnx_opts():
    if not HAS_LAB or settings.SIMULATION_MODE:
        return [
            "eliminate_identity",
            "fuse_add_bias_into_conv",
            "fuse_transpose_into_gemm",
            "eliminate_unused_initializer"
        ]
    lab = ONNXCompilerLab()
    return lab.list_optimizations()

@router.post("/onnx/analyze")
async def analyze_onnx(model_path: str):
    if not HAS_LAB or settings.SIMULATION_MODE:
        return {
            "op_counts": {"Conv": 50, "Relu": 50, "Gemm": 10},
            "total_nodes": 110,
            "optimization_score": 0.85
        }
    lab = ONNXCompilerLab(model_path)
    return lab.analyze_graph()

@router.get("/observatory/weights")
async def observe_weights():
    if not HAS_LAB or settings.SIMULATION_MODE:
        return [
            {"layer": "self_attn.q_proj", "issue": "High weight variance", "recommendation": "Per-channel quantization"},
            {"layer": "mlp.gate_proj", "issue": "High activation outliers", "recommendation": "SmoothQuant"}
        ]
    obs = ModelInternalsObservatory()
    return obs.detect_bottlenecks()
