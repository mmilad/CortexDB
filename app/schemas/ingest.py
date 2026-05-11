"""Schemas for pipeline-backed ingest adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.processor import ProcessorStrategy


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to chunk and ingest.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller metadata merged into each generated chunk.",
    )
    max_chars: int = Field(default=2000, ge=1)
    overlap_chars: int = Field(default=200, ge=0)
    ingestion_id: str | None = Field(default=None)
    batch_size: int = Field(default=100, ge=1, le=500)
    processor_strategy: ProcessorStrategy = Field(
        default="fallback",
        description=(
            "Optional processor strategy. 'fallback' uses the built-in deterministic "
            "paragraph chunker; other values require an enabled processor service."
        ),
    )
    extract_primitives: bool = Field(
        default=False,
        description="When true, store processor-extracted meaning primitives as derived items.",
    )
