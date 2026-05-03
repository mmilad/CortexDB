"""Memory item ingest and search endpoints.

POST /datasets/{key}/ingest  — batch ingest raw text items; CortexDB vectorizes them.
POST /datasets/{key}/search  — raw text query → embed → cosine similarity → top-k.
GET  /datasets/{key}/items   — list stored items (paginated).
GET  /datasets/{key}/items/{id} — get one item.
DELETE /datasets/{key}/items/{id} — delete one item.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.embed.service import EmbeddingService, get_embedding_service
from app.schemas.memory import (
    IngestRequest,
    IngestResult,
    MemoryItem,
    SearchRequest,
    SearchResponse,
    SearchHit,
)
from app.store import SqliteStore, get_store

router = APIRouter(tags=["memory"])


def _row_to_memory_item(row: dict) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        dataset_key=row["dataset_key"],
        raw_text=row["raw_text"],
        metadata=row["metadata"],
        embedding_model=row.get("embedding_model"),
        created_at=row.get("created_at"),
    )


@router.post(
    "/datasets/{dataset_key}/ingest",
    response_model=IngestResult,
    summary="Ingest raw text items into a dataset",
    description=(
        "Accepts raw text items. CortexDB embeds each item using the configured "
        "embedding provider (default: nomic-embed-text via Ollama) and stores "
        "both the raw text and the vector. Callers never send pre-computed vectors. "
        "Re-embedding is possible later because the raw text is always preserved."
    ),
)
def ingest_items(
    dataset_key: str,
    body: IngestRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> IngestResult:
    if not store.get_dataset(dataset_key):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")

    if not embed_svc.is_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
                "Set CORTEXDB_EMBED_PROVIDER=ollama or =api to enable ingest."
            ),
        )

    texts = [item.raw_text for item in body.items]
    vectors = embed_svc.embed(texts)
    model_id = embed_svc.model_id

    ids = []
    for item, vector in zip(body.items, vectors):
        item_id = item.id or str(uuid.uuid4())
        store.insert_memory_item(
            {
                "id": item_id,
                "dataset_key": dataset_key,
                "raw_text": item.raw_text,
                "metadata": item.metadata,
                "embedding": vector,
                "embedding_model": model_id,
            }
        )
        ids.append(item_id)

    return IngestResult(ingested=len(ids), ids=ids, embedding_model=model_id)


@router.post(
    "/datasets/{dataset_key}/search",
    response_model=SearchResponse,
    summary="Semantic search over a dataset's memory items",
    description=(
        "Embeds the raw query text using the configured provider, then runs "
        "cosine similarity over stored items. Optionally filter by metadata "
        "key-value pairs before scoring. Returns top-k hits with scores."
    ),
)
def search_items(
    dataset_key: str,
    body: SearchRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> SearchResponse:
    if not store.get_dataset(dataset_key):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")

    if not embed_svc.is_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
                "Set CORTEXDB_EMBED_PROVIDER=ollama or =api to enable search."
            ),
        )

    query_vector = embed_svc.embed_one(body.query)
    model_id = embed_svc.model_id

    raw_results = store.search_memory_items(
        dataset_key=dataset_key,
        query_vector=query_vector,
        top_k=body.top_k,
        metadata_filters=body.metadata_filters,
    )

    total_searched = store.count_memory_items(dataset_key)

    hits = []
    for row in raw_results:
        if row["score"] < body.min_score:
            continue
        hits.append(
            SearchHit(
                item=_row_to_memory_item(row),
                score=row["score"],
            )
        )

    return SearchResponse(
        hits=hits,
        query=body.query,
        embedding_model=model_id,
        total_searched=total_searched,
    )


@router.get(
    "/datasets/{dataset_key}/items",
    response_model=list[MemoryItem],
    summary="List memory items in a dataset",
)
def list_items(
    dataset_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[MemoryItem]:
    if not store.get_dataset(dataset_key):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")
    rows = store.list_memory_items(dataset_key, limit=limit, offset=offset)
    return [_row_to_memory_item(r) for r in rows]


@router.get(
    "/datasets/{dataset_key}/items/{item_id}",
    response_model=MemoryItem,
    summary="Get a single memory item",
)
def get_item(
    dataset_key: str,
    item_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> MemoryItem:
    row = store.get_memory_item(item_id)
    if not row or row["dataset_key"] != dataset_key:
        raise HTTPException(status_code=404, detail="item not found")
    return _row_to_memory_item(row)


@router.delete(
    "/datasets/{dataset_key}/items/{item_id}",
    summary="Delete a memory item",
)
def delete_item(
    dataset_key: str,
    item_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict:
    row = store.get_memory_item(item_id)
    if not row or row["dataset_key"] != dataset_key:
        raise HTTPException(status_code=404, detail="item not found")
    store.delete_memory_item(item_id)
    return {"deleted": item_id}
