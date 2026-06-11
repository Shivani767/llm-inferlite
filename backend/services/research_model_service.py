from sqlalchemy.orm import Session

from database.repositories.model_repository import ModelRegistryRepository
from services.model_service import ModelRegistryService


class ResearchModelService(ModelRegistryService):
    def __init__(self, db: Session):
        super().__init__(ModelRegistryRepository(db))
