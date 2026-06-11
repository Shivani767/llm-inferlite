from sqlalchemy.orm import Session, sessionmaker

from database.repositories.model_repository import ModelRegistryRepository
from services.model_service import ModelRegistryService


class AppContainer:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def create_db_session(self) -> Session:
        return self.session_factory()

    def get_model_registry_service(self, db_session: Session) -> ModelRegistryService:
        repository = ModelRegistryRepository(db_session)
        return ModelRegistryService(repository)
