from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = BASE_DIR / "inferlite.db"


class Settings(BaseSettings):
    PROJECT_NAME: str = "InferLite"
    PROJECT_DESCRIPTION: str = "The local research operating system for LLM inference optimization."
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "local"
    LOG_LEVEL: str = "INFO"
    ENABLE_OTEL: bool = True

    DATABASE_URL: str = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    REDIS_URL: str = "redis://localhost:6379/0"

    # HF Settings
    HF_TOKEN: Optional[str] = None
    MODELS_DIR: str = str(BASE_DIR / "data" / "models")

    # Celery Settings
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # Simulation Settings
    SIMULATION_MODE: bool = True

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = self.REDIS_URL
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL


settings = Settings()
