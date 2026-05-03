from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.embed.service import EmbeddingService, get_embedding_service
from app.schemas.dataset import DatasetRecord
from app.schemas.discovery import DatasetDiscoverRequest, DatasetDiscoverResponse
from app.services.dataset_match import discover_datasets
from app.store import SqliteStore, get_store

logger = logging.getLogger("cortexdb.datasets")

router = APIRouter(tags=["datasets"])


def _embed_text_for_dataset(record: DatasetRecord) -> str:
    """Compose the text CortexDB embeds to represent this dataset.

    Combines the most semantically rich fields so the dataset's vector
    captures its purpose, usage, and entity types.
    """
    parts = [record.display_name]
    if record.llm_summary:
        parts.append(record.llm_summary)
    parts.append(record.semantic_description)
    parts.append(record.usage_guidance)
    if record.entity_types:
        parts.append("Entities: " + ", ".join(record.entity_types))
    if record.access_patterns:
        parts.append("Patterns: " + ", ".join(record.access_patterns))
    if record.capability_tags:
        parts.append("Tags: " + ", ".join(record.capability_tags))
    return " | ".join(parts)


@router.post(
    "/datasets/discover",
    response_model=DatasetDiscoverResponse,
    summary="Match or propose a dataset",
    description=(
        "Finds the best matching datasets for a given intent. "
        "When embedding is enabled, uses vector similarity over dataset descriptions "
        "combined with deterministic capability/tag scoring. "
        "Falls back to token-overlap scoring when embedding is unavailable."
    ),
)
def post_discover_datasets(
    body: DatasetDiscoverRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DatasetDiscoverResponse:
    # Attempt to embed the intent for vector-boosted discovery
    intent_vector: list[float] | None = None
    if embed_svc.is_enabled():
        try:
            intent_vector = embed_svc.embed_one(body.intent)
        except Exception as exc:
            logger.warning("Could not embed intent for discovery: %s", exc)

    return discover_datasets(body, store, intent_vector)


@router.post("/datasets", response_model=DatasetRecord)
def upsert_dataset(
    record: DatasetRecord,
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DatasetRecord:
    store.upsert_dataset(record.dataset_key, record.model_dump())

    # Auto-embed the dataset description so it is discoverable via vector search.
    if embed_svc.is_enabled():
        try:
            raw_text = _embed_text_for_dataset(record)
            vector = embed_svc.embed_one(raw_text)
            store.set_dataset_embedding(record.dataset_key, raw_text, vector, embed_svc.model_id)
        except Exception as exc:
            # Log and continue — embedding failure does not block registration.
            logger.warning(
                "Could not embed dataset '%s': %s. "
                "Dataset is registered but will not appear in vector discovery.",
                record.dataset_key,
                exc,
            )

    return record


@router.get("/datasets", response_model=list[DatasetRecord])
def list_datasets(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> list[DatasetRecord]:
    return [DatasetRecord(**d) for d in store.list_datasets().values()]


@router.get("/datasets/{dataset_key}", response_model=DatasetRecord)
def get_dataset(
    dataset_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> DatasetRecord:
    data = store.get_dataset(dataset_key)
    if not data:
        raise HTTPException(status_code=404, detail="dataset not found")
    return DatasetRecord(**data)
