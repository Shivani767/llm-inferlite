from sqlalchemy.orm import Session
from database.models import Experiment
from typing import Dict, Any, List, Optional
from datetime import datetime

class ExperimentTracker:
    """
    Module 13: Systematic tracking of research experiments.
    Ensures reproducibility by capturing the full software/hardware stack.
    """
    
    def __init__(self, db: Session):
        self.db = db

    def log_experiment(
        self, 
        name: str, 
        cuda_version: str, 
        driver_version: str, 
        hardware: str, 
        config: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Experiment:
        """
        Records an experiment with all its environmental metadata.
        """
        experiment = Experiment(
            name=name,
            cuda_version=cuda_version,
            driver_version=driver_version,
            hardware=hardware,
            config=config,
            results=results,
            created_at=datetime.utcnow()
        )
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment

    def get_experiments(self, skip: int = 0, limit: int = 100) -> List[Experiment]:
        return self.db.query(Experiment).offset(skip).limit(limit).all()

    def get_experiment_by_id(self, experiment_id: int) -> Optional[Experiment]:
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
