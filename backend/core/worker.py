from celery import Celery
from core.config import settings

celery_app = Celery(
    "inferlite_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["workers.tasks"],
)

celery_app.conf.task_routes = {
    "worker.tasks.quantize_model": "quantization-queue",
    "worker.tasks.run_benchmark": "benchmark-queue",
}

@celery_app.task
def test_task(name: str):
    return f"Hello {name}, Celery is working!"
