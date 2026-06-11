from typing import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from core.container import AppContainer
from services.model_service import ModelRegistryService


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_db_session(
    container: AppContainer = Depends(get_container),
) -> Generator[Session, None, None]:
    db_session = container.create_db_session()
    try:
        yield db_session
    finally:
        db_session.close()


def get_model_registry_service(
    container: AppContainer = Depends(get_container),
    db_session: Session = Depends(get_db_session),
) -> ModelRegistryService:
    return container.get_model_registry_service(db_session)
