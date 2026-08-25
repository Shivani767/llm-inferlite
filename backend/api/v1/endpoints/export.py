from fastapi import APIRouter
try:
    from exporters.engine import ExportEngine
    HAS_EXPORTER = True
except ImportError:
    HAS_EXPORTER = False
from pydantic import BaseModel

router = APIRouter()

class ExportRequest(BaseModel):
    model_version_id: int
    target_format: str

@router.post("/")
async def export_model(request: ExportRequest):
    engine = ExportEngine()
    if request.target_format == "onnx":
        return await engine.export_to_onnx(str(request.model_version_id), "path/to/output")
    elif request.target_format == "gguf":
        return await engine.export_to_gguf(str(request.model_version_id), "path/to/output")
    
    return {"status": "error", "message": "Unsupported format"}
