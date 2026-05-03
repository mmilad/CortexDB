"""Memory item ingest, search, and lifecycle endpoints.

POST /datasets/{key}/ingest         — batch ingest raw text items; CortexDB vectorizes.
POST /datasets/{key}/search         — vector, keyword, or hybrid search → top-k hits.
POST /datasets/{key}/re-embed       — re-vectorize all items with current embed model.
GET  /datasets/{key}/items          — list stored items (paginated; skip soft-deleted).
GET  /datasets/{key}/items/{id}     — get one item.
DELETE /datasets/{key}/items/{id}   — soft-delete (recoverable).
DELETE /datasets/{key}/items/{id}/hard — hard-delete (irreversible).
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
    SearchHit,
    SearchRequest,
    SearchResponse,
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
        is_deleted=row.get("is_deleted", False),
    )


def _resolve_search_mode(body: "SearchRequest") -> str:
    has_vector = body.vector_weight > 0.0
    has_keyword = body.keyword_query is not None
    if has_vector and has_keyword:
        return "hybrid"
    if has_keyword:
        return "keyword"
    return "vector"


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

    search_mode = _resolve_search_mode(body)
    needs_embedding = search_mode in ("vector", "hybrid")

    if needs_embedding and not embed_svc.is_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
                "For vector or hybrid search set CORTEXDB_EMBED_PROVIDER=ollama or =api. "
                "For keyword-only search set vector_weight=0.0 and provide keyword_query."
            ),
        )

    query_vector: list[float] | None = None
    model_id: str | None = None
    if needs_embedding:
        query_vector = embed_svc.embed_one(body.query)
        model_id = embed_svc.model_id

    raw_results = store.search_memory_items(
        dataset_key=dataset_key,
        query_vector=query_vector,
        top_k=body.top_k,
        metadata_filters=body.metadata_filters,
        keyword_query=body.keyword_query,
        vector_weight=body.vector_weight,
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
                vector_score=row.get("vector_score"),
                keyword_score=row.get("keyword_score"),
            )
        )

    return SearchResponse(
        hits=hits,
        query=body.query,
        embedding_model=model_id,
        total_searched=total_searched,
        search_mode=search_mode,
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
    include_deleted: bool = Query(
        default=False,
        description="When true, soft-deleted items are included in the response.",
    ),
) -> list[MemoryItem]:
    if not store.get_dataset(dataset_key):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")
    rows = store.list_memory_items(dataset_key, limit=limit, offset=offset, include_deleted=include_deleted)
    return [_row_to_memory_item(r) for r in rows]


@router.get(
    "/datasets/{dataset_key}/items/{item_id}",
    response_model=MemoryItem,
    summary="Get a single memory item",
    description=(
        "Returns the item. Soft-deleted items return 404 unless include_deleted=true is passed."
    ),
)
def get_item(
    dataset_key: str,
    item_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    include_deleted: bool = Query(
        default=False,
        description="When true, a soft-deleted item is returned instead of 404.",
    ),
) -> MemoryItem:
    row = store.get_memory_item(item_id)
    if not row or row["dataset_key"] != dataset_key:
        raise HTTPException(status_code=404, detail="item not found")
    if row.get("is_deleted") and not include_deleted:
        raise HTTPException(status_code=404, detail="item not found")
    return _row_to_memory_item(row)


@router.delete(
    "/datasets/{dataset_key}/items/{item_id}",
    summary="Soft-delete a memory item",
    description=(
        "Marks the item as deleted. It is excluded from all queries by default "
        "but can be retrieved using include_deleted=true on the list endpoint. "
        "Use DELETE .../hard to permanently remove the row."
    ),
)
def delete_item(
    dataset_key: str,
    item_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict:
    row = store.get_memory_item(item_id)
    if not row or row["dataset_key"] != dataset_key:
        raise HTTPException(status_code=404, detail="item not found")
    deleted = store.soft_delete_memory_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="item not found or already deleted")
    return {"soft_deleted": item_id}


@router.delete(
    "/datasets/{dataset_key}/items/{item_id}/hard",
    summary="Hard-delete a memory item",
    description="Permanently removes the memory item row from the database. Irreversible.",
)
def hard_delete_item(
    dataset_key: str,
    item_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict:
    row = store.get_memory_item(item_id)
    if not row or row["dataset_key"] != dataset_key:
        raise HTTPException(status_code=404, detail="item not found")
    store.delete_memory_item(item_id)
    return {"hard_deleted": item_id}


@router.post(
    "/datasets/{dataset_key}/re-embed",
    summary="Re-embed all items in a dataset",
    description=(
        "Re-vectorizes every memory item in the dataset using the currently configured "
        "embedding provider. Use this after changing the embedding model to keep all "
        "vectors on the same model version. "
        "Items are processed in batches. Returns counts of updated and failed items."
    ),
)
def re_embed_dataset(
    dataset_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
    batch_size: int = Query(default=50, ge=1, le=500, description="Items per embedding batch."),
) -> dict:
    if not store.get_dataset(dataset_key):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")

    if not embed_svc.is_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
                "Set CORTEXDB_EMBED_PROVIDER=ollama or =api to enable re-embedding."
            ),
        )

    items = store.list_all_memory_items(dataset_key)
    model_id = embed_svc.model_id

    updated = 0
    failed = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [it["raw_text"] for it in batch]
        try:
            vectors = embed_svc.embed(texts)
        except Exception:
            failed += len(batch)
            continue

        for item, vector in zip(batch, vectors):
            try:
                store.update_memory_item_embedding(item["id"], vector, model_id)
                updated += 1
            except Exception:
                failed += 1

    # Rebuild the ANN index now that embeddings (and possibly the dimension) have changed.
    vec_indexed = store.rebuild_vec_index(dataset_key)

    return {
        "dataset_key": dataset_key,
        "embedding_model": model_id,
        "total_items": len(items),
        "updated": updated,
        "failed": failed,
        "vec_indexed": vec_indexed,
    }
