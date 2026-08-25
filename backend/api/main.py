from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from api.v1 import api_router
from core.logging import setup_logging
from core.config import settings
from core.container import AppContainer
from database.session import check_database_connection, init_db
from database.session import SessionLocal

logger = logging.getLogger(__name__)


def configure_telemetry() -> None:
    if not (HAS_OTEL and settings.ENABLE_OTEL):
        return

    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging(settings.LOG_LEVEL)
    configure_telemetry()
    init_db()
    logger.info("InferLite startup complete")
    yield
    logger.info("InferLite shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.container = AppContainer(SessionLocal)

    if HAS_OTEL and settings.ENABLE_OTEL:
        FastAPIInstrumentor.instrument_app(app)

    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/")
    async def root():
        return {
            "message": "Welcome to InferLite API",
            "project": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "database": "up" if check_database_connection() else "down",
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
        }

    return app


app = create_app()
