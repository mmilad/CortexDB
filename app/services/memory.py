"""Reusable memory ingest service.

This module owns the deterministic storage path for raw text ingest. API
routes and higher-level pipelines should call this instead of duplicating
embedding or vector insertion logic.
"""

from __future__ import annotations

import uuid

from app.embed.service import EmbeddingService
from app.schemas.memory import IngestItem, IngestResult
from app.store import SqliteStore


class DatasetNotFoundError(ValueError):
    """Raised when an ingest target dataset is missing."""


class EmbeddingDisabledError(RuntimeError):
    """Raised when ingest requires embeddings but embedding is disabled."""


async def ingest_items_to_dataset(
    dataset_key: str,
    items: list[IngestItem],
    store: SqliteStore,
    embed_svc: EmbeddingService,
) -> IngestResult:
    """Embed and store raw text items in an existing dataset."""
    if not store.get_dataset(dataset_key):
        raise DatasetNotFoundError(f"Dataset '{dataset_key}' not found")

    if not embed_svc.is_enabled():
        raise EmbeddingDisabledError(
            "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
            "Set CORTEXDB_EMBED_PROVIDER=ollama or =api to enable ingest."
        )

    if not items:
        return IngestResult(ingested=0, ids=[], embedding_model=embed_svc.model_id)

    texts = [item.raw_text for item in items]
    vectors = await embed_svc.embed(texts)
    model_id = embed_svc.model_id

    ids: list[str] = []
    for item, vector in zip(items, vectors):
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
