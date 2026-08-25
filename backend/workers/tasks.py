from core.worker import celery_app
from quantization.engine import QuantizationEngine
from database.session import SessionLocal
from database.repositories.model_repository import ModelRegistryRepository

@celery_app.task(name="worker.tasks.quantize_model")
def quantize_model_task(job_id: int):
    db = SessionLocal()
    repo = ModelRegistryRepository(db)
    
    # In a real app, we would fetch the job from the DB
    # For now, this is a placeholder for the logic
    
    engine = QuantizationEngine()
    
    # Simulate quantization
    # loop = asyncio.get_event_loop()
    # result = loop.run_until_complete(engine.run_quantization(...))
    
    print(f"Running quantization for job {job_id}")
    
    db.close()
    return {"status": "completed", "job_id": job_id}

@celery_app.task(name="worker.tasks.run_benchmark")
def run_benchmark_task(benchmark_id: int):
    print(f"Running benchmark {benchmark_id}")
    return {"status": "completed", "benchmark_id": benchmark_id}
