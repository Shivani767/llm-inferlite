from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.session import get_db
try:
    from workers.tasks import quantize_model_task
    HAS_WORKER = True
except ImportError:
    HAS_WORKER = False
from pydantic import BaseModel
from typing import Dict, Any, Optional

router = APIRouter()

class QuantizationRequest(BaseModel):
    model_version_id: int
    target_format: str
    target_quantization: str
    config: Optional[Dict[str, Any]] = {}

@router.post("/")
async def start_quantization(
    request: QuantizationRequest,
    db: Session = Depends(get_db)
):
    # Here we would create a QuantizationJob in the DB
    # For now, we just trigger the task
    
    task = quantize_model_task.delay(job_id=123) # Dummy ID
    
    return {
        "job_id": 123,
        "task_id": task.id,
        "status": "queued"
    }
