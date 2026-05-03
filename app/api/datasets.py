from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.dataset import DatasetRecord
from app.schemas.discovery import DatasetDiscoverRequest, DatasetDiscoverResponse
from app.services.dataset_match import discover_datasets
from app.state import RegistryState, get_registry

router = APIRouter(tags=["datasets"])


@router.post(
    "/datasets/discover",
    response_model=DatasetDiscoverResponse,
    summary="Match or propose a dataset",
    description=(
        "Deterministic registry matching over metadata (capabilities, tags, token overlap). "
        "Semantic vector similarity over descriptions is not used in v1; callers may add "
        "registry embeddings in a later release."
    ),
)
def post_discover_datasets(
    body: DatasetDiscoverRequest,
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> DatasetDiscoverResponse:
    return discover_datasets(body, reg.datasets)


@router.post("/datasets", response_model=DatasetRecord)
def upsert_dataset(
    record: DatasetRecord,
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> DatasetRecord:
    reg.datasets[record.dataset_key] = record.model_dump()
    return record


@router.get("/datasets", response_model=list[DatasetRecord])
def list_datasets(
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> list[DatasetRecord]:
    return [DatasetRecord(**d) for d in reg.datasets.values()]


@router.get("/datasets/{dataset_key}", response_model=DatasetRecord)
def get_dataset(
    dataset_key: str,
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> DatasetRecord:
    data = reg.datasets.get(dataset_key)
    if not data:
        raise HTTPException(status_code=404, detail="dataset not found")
    return DatasetRecord(**data)
