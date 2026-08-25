from typing import List, Optional

from sqlalchemy.orm import Session

from database.models import Model, ModelVersion
from models.schemas_research import ModelCreate, ModelVersionCreate


class SQLAlchemyModelRegistryRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id: int) -> Optional[Model]:
        return self.db.query(Model).filter(Model.id == model_id).first()

    def get_by_name(self, name: str) -> Optional[Model]:
        return self.db.query(Model).filter(Model.name == name).first()

    def list_models(self, skip: int = 0, limit: int = 100) -> List[Model]:
        return self.db.query(Model).offset(skip).limit(limit).all()

    def create_model(self, model: ModelCreate) -> Model:
        db_model = Model(**model.model_dump())
        self.db.add(db_model)
        self.db.commit()
        self.db.refresh(db_model)
        return db_model

    def add_version(self, version: ModelVersionCreate) -> ModelVersion:
        db_version = ModelVersion(**version.model_dump())
        self.db.add(db_version)
        self.db.commit()
        self.db.refresh(db_version)
        return db_version

    def get_version_by_sha256(self, sha256: str) -> Optional[ModelVersion]:
        return self.db.query(ModelVersion).filter(ModelVersion.sha256 == sha256).first()


ModelRegistryRepository = SQLAlchemyModelRegistryRepository
ModelRepository = SQLAlchemyModelRegistryRepository
