import os
import hashlib
from typing import Optional

from core.config import settings
from models.schemas_research import ModelCreate, ModelImportRequest, ModelVersionCreate
from repositories.interfaces import ModelRegistryRepositoryProtocol

try:
    from huggingface_hub import HfApi, model_info

    HAS_HF = True
except ImportError:
    HAS_HF = False


class ModelRegistryService:
    def __init__(self, repository: ModelRegistryRepositoryProtocol):
        self.repository = repository
        self.hf_api = HfApi(token=settings.HF_TOKEN) if HAS_HF else None

    def list_models(self, skip: int = 0, limit: int = 100):
        return self.repository.list_models(skip=skip, limit=limit)

    def get_model(self, model_id: int):
        return self.repository.get_by_id(model_id)

    async def import_model(self, request: ModelImportRequest):
        if request.source_type == "huggingface":
            return await self._import_from_hf(request)
        if request.source_type == "local":
            return await self._import_from_local(request)
        raise ValueError(f"Unsupported source type: {request.source_type}")

    async def _import_from_hf(self, request: ModelImportRequest):
        if not HAS_HF:
            raise ValueError("HuggingFace dependencies not installed")

        info = model_info(request.repo_id, revision=request.version_tag)
        config = getattr(info, "config", {}) or {}
        architecture = config.get("model_type", "unknown")
        context_length = config.get("max_position_embeddings") or config.get("n_ctx")

        model = self.repository.get_by_name(request.repo_id)
        if not model:
            model = self.repository.create_model(
                ModelCreate(
                    name=request.repo_id,
                    family=self._determine_family(architecture, request.repo_id),
                    architecture=architecture,
                    parameters=self._estimate_parameters(info),
                    context_length=context_length,
                    tokenizer_type=config.get("tokenizer_class", "unknown"),
                )
            )

        total_size = sum(
            sibling.size
            for sibling in getattr(info, "siblings", [])
            if hasattr(sibling, "size") and sibling.size
        )
        self.repository.add_version(
            ModelVersionCreate(
                model_id=model.id,
                version_tag=request.version_tag,
                format=request.format,
                size_bytes=total_size or None,
                metadata_json={"repo_id": request.repo_id, "config": config},
            )
        )
        return self.repository.get_by_id(model.id)

    async def _import_from_local(self, request: ModelImportRequest):
        if not os.path.exists(request.repo_id):
            raise FileNotFoundError(f"Local path {request.repo_id} not found")

        model_name = os.path.basename(request.repo_id.rstrip("\\/"))
        model = self.repository.get_by_name(model_name)
        if not model:
            model = self.repository.create_model(
                ModelCreate(
                    name=model_name,
                    family="local",
                    architecture="unknown",
                )
            )

        sha256 = self._calculate_sha256(request.repo_id) if os.path.isfile(request.repo_id) else None
        if sha256 and self.repository.get_version_by_sha256(sha256):
            return self.repository.get_by_id(model.id)

        self.repository.add_version(
            ModelVersionCreate(
                model_id=model.id,
                version_tag=request.version_tag,
                format=request.format,
                sha256=sha256,
                size_bytes=os.path.getsize(request.repo_id) if os.path.isfile(request.repo_id) else 0,
                metadata_json={"source_type": "local", "path": request.repo_id},
            )
        )
        return self.repository.get_by_id(model.id)

    def _determine_family(self, architecture: str, name: str) -> str:
        lowered_name = name.lower()
        lowered_architecture = architecture.lower()
        if "llama" in lowered_name or lowered_architecture == "llama":
            return "Llama"
        if "mistral" in lowered_name or lowered_architecture == "mistral":
            return "Mistral"
        if "qwen" in lowered_name or lowered_architecture == "qwen":
            return "Qwen"
        if "gemma" in lowered_name or lowered_architecture == "gemma":
            return "Gemma"
        if "deepseek" in lowered_name:
            return "DeepSeek"
        return "Other"

    def _estimate_parameters(self, info) -> str:
        for tag in getattr(info, "tags", []):
            if tag.endswith("b") and tag[:-1].replace(".", "").isdigit():
                return tag.upper()
        return "Unknown"

    def _calculate_sha256(self, path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as file_handle:
            for byte_block in iter(lambda: file_handle.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


ModelService = ModelRegistryService
