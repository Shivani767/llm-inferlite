from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from database.models import ModelFormat

class ModelVersionBase(BaseModel):
    version_tag: str
    format: ModelFormat
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata_json: Optional[Dict[str, Any]] = None

class ModelVersionCreate(ModelVersionBase):
    model_id: int

class ModelVersion(ModelVersionBase):
    id: int
    model_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ModelBase(BaseModel):
    name: str
    family: Optional[str] = None
    architecture: Optional[str] = None
    parameters: Optional[str] = None
    context_length: Optional[int] = None
    tokenizer_type: Optional[str] = None

class ModelCreate(ModelBase):
    pass

class Model(ModelBase):
    id: int
    created_at: datetime
    versions: List[ModelVersion] = []
    model_config = ConfigDict(from_attributes=True)

class ModelImportRequest(BaseModel):
    repo_id: str # HuggingFace repo ID or local path
    version_tag: str = "main"
    format: ModelFormat = ModelFormat.HF
    source_type: str = "huggingface" # huggingface, local
