from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.session import get_db
from experiments.tracker import ExperimentTracker
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class ExperimentLog(BaseModel):
    name: str
    cuda_version: str
    driver_version: str
    hardware: str
    config: Dict[str, Any]
    results: Dict[str, Any]

@router.post("/log", status_code=201)
async def log_research_experiment(
    experiment: ExperimentLog, 
    db: Session = Depends(get_db)
):
    """
    Module 13: Log a research experiment for tracking and reproducibility.
    """
    tracker = ExperimentTracker(db)
    return tracker.log_experiment(
        experiment.name,
        experiment.cuda_version,
        experiment.driver_version,
        experiment.hardware,
        experiment.config,
        experiment.results
    )

@router.get("/", response_model=List[Dict[str, Any]])
async def list_experiments(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    return tracker.get_experiments(skip, limit)
