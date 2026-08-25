from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_model_registry_service
from models.schemas_research import Model, ModelImportRequest
from services.model_service import ModelRegistryService

router = APIRouter()


@router.get("/", response_model=List[Model])
def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=500),
    service: ModelRegistryService = Depends(get_model_registry_service),
):
    return service.list_models(skip=skip, limit=limit)


@router.get("/{model_id}", response_model=Model)
def get_model(
    model_id: int,
    service: ModelRegistryService = Depends(get_model_registry_service),
):
    model = service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/import", response_model=Model)
async def import_model(
    request: ModelImportRequest,
    service: ModelRegistryService = Depends(get_model_registry_service),
):
    try:
        return await service.import_model(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
