from __future__ import annotations

from typing import Optional, Protocol, Sequence

from database.models import Model, ModelVersion
from models.schemas_research import ModelCreate, ModelVersionCreate


class ModelRegistryRepositoryProtocol(Protocol):
    def get_by_id(self, model_id: int) -> Optional[Model]:
        ...

    def get_by_name(self, name: str) -> Optional[Model]:
        ...

    def list_models(self, skip: int = 0, limit: int = 100) -> Sequence[Model]:
        ...

    def create_model(self, model: ModelCreate) -> Model:
        ...

    def add_version(self, version: ModelVersionCreate) -> ModelVersion:
        ...

    def get_version_by_sha256(self, sha256: str) -> Optional[ModelVersion]:
        ...
