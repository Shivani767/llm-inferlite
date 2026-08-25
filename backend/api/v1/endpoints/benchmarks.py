from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.session import get_db
try:
    from workers.tasks import run_benchmark_task
    HAS_WORKER = True
except ImportError:
    HAS_WORKER = False
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from benchmarking.benchmark_farm import BenchmarkFarm

router = APIRouter()

class BenchmarkRequest(BaseModel):
    model_version_id: int
    scenarios: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = {}

@router.post("/farm/auto-process/{model_id}")
async def trigger_benchmark_farm(
    model_id: int,
    db: Session = Depends(get_db)
):
    """
    Automated 'Quantize -> Benchmark -> Evaluate' pipeline.
    Typically triggered after a new model is added to the registry.
    """
    farm = BenchmarkFarm(db)
    try:
        results = await farm.process_new_model(model_id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{benchmark_id}")
async def get_benchmark_results(benchmark_id: int, db: Session = Depends(get_db)):
    # Fetch results from DB
    return {"id": benchmark_id, "status": "completed", "results": {}}
