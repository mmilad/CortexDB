"""Reusable deterministic ingest pipeline service."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.embed.service import EmbeddingService
from app.ingest.chunking import chunk_text
from app.ingest.sources import SourceDocument, iter_source_documents
from app.schemas.memory import IngestItem, IngestResult
from app.services.memory import ingest_items_to_dataset
from app.store import SqliteStore

DEFAULT_MAX_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_BATCH_SIZE = 100


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_digest(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return _sha256(raw)


def _default_ingestion_id(
    docs: list[SourceDocument],
    *,
    max_chars: int,
    overlap_chars: int,
) -> str:
    source_digest = hashlib.sha256()
    for doc in docs:
        source_digest.update(doc.source_type.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(doc.source_identity.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(_sha256(doc.text).encode("ascii"))
        source_digest.update(b"\0")
    source_digest.update(str(max_chars).encode("ascii"))
    source_digest.update(b"\0")
    source_digest.update(str(overlap_chars).encode("ascii"))
    return f"ingest-{source_digest.hexdigest()[:16]}"


def _build_doc_items(
    doc: SourceDocument,
    *,
    metadata: dict[str, Any],
    ingestion_id: str,
    max_chars: int,
    overlap_chars: int,
) -> list[IngestItem]:
    chunks = chunk_text(doc.text, max_chars=max_chars, overlap_chars=overlap_chars)
    source_sha256 = _sha256(doc.text)
    chunk_count = len(chunks)
    items: list[IngestItem] = []

    for chunk_index, raw_text in enumerate(chunks):
        content_sha256 = _sha256(raw_text)
        item_metadata = {
            **metadata,
            "source_type": doc.source_type,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "content_sha256": content_sha256,
            "source_sha256": source_sha256,
            "ingestion_id": ingestion_id,
        }
        if doc.source_path is not None:
            item_metadata["source_path"] = doc.source_path
        if doc.filename is not None:
            item_metadata["filename"] = doc.filename

        item_id = "ingest-" + _stable_digest(
            doc.source_identity,
            source_sha256,
            max_chars,
            overlap_chars,
            chunk_index,
            content_sha256,
        )[:32]
        items.append(IngestItem(id=item_id, raw_text=raw_text, metadata=item_metadata))

    return items


def build_ingest_items(
    source: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    metadata: dict[str, Any] | None = None,
    ingestion_id: str | None = None,
) -> list[IngestItem]:
    """Convert a text source, file, or directory into CortexDB ingest items."""
    docs = iter_source_documents(source)
    resolved_ingestion_id = ingestion_id or _default_ingestion_id(
        docs,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    items: list[IngestItem] = []
    base_metadata = dict(metadata or {})
    for doc in docs:
        items.extend(
            _build_doc_items(
                doc,
                metadata=base_metadata,
                ingestion_id=resolved_ingestion_id,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return items


async def ingest_source_to_dataset(
    dataset_key: str,
    source: str | Path,
    store: SqliteStore,
    embed_svc: EmbeddingService,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    metadata: dict[str, Any] | None = None,
    ingestion_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestResult:
    """Build and ingest source chunks into a dataset using existing ingest logic."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    items = build_ingest_items(
        source,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        metadata=metadata,
        ingestion_id=ingestion_id,
    )

    ids: list[str] = []
    embedding_model: str | None = None
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        result = await ingest_items_to_dataset(dataset_key, batch, store, embed_svc)
        ids.extend(result.ids)
        embedding_model = result.embedding_model

    return IngestResult(ingested=len(ids), ids=ids, embedding_model=embedding_model)


async def ingest_directory_to_dataset(
    dataset_key: str,
    directory_path: str | Path,
    store: SqliteStore,
    embed_svc: EmbeddingService,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    metadata: dict[str, Any] | None = None,
    ingestion_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestResult:
    """Ingest all supported files in a directory tree."""
    path = Path(directory_path)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    return await ingest_source_to_dataset(
        dataset_key,
        path,
        store,
        embed_svc,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        metadata=metadata,
        ingestion_id=ingestion_id,
        batch_size=batch_size,
    )
