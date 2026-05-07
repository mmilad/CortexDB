"""Deterministic source-to-IngestItem pipeline for CortexDB."""

from app.ingest.chunking import chunk_text
from app.ingest.service import (
    build_ingest_items,
    ingest_directory_to_dataset,
    ingest_source_to_dataset,
)

__all__ = [
    "build_ingest_items",
    "chunk_text",
    "ingest_directory_to_dataset",
    "ingest_source_to_dataset",
]
