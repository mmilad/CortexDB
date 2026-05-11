"""Deterministic source-to-IngestItem pipeline for CortexDB."""

from app.ingest.analyzer import analyze_ingest
from app.ingest.chunking import chunk_text
from app.ingest.service import (
    build_ingest_items,
    build_ingest_items_with_processor,
    ingest_directory_to_dataset,
    ingest_source_to_dataset,
)

__all__ = [
    "analyze_ingest",
    "build_ingest_items",
    "build_ingest_items_with_processor",
    "chunk_text",
    "ingest_directory_to_dataset",
    "ingest_source_to_dataset",
]
