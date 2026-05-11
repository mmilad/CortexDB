"""Schemas for memory items — the unit of storable + searchable content in CortexDB."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.processor import ProcessorJobResult


class IngestItem(BaseModel):
    """One text item to ingest into a dataset.

    Callers supply raw_text only. CortexDB handles vectorization.
    """

    id: str | None = Field(
        default=None,
        description="Optional stable id. A UUID is generated if omitted.",
    )
    raw_text: str = Field(
        ...,
        description="The raw text to store and vectorize. Never send pre-computed vectors.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value metadata attached to this item. "
            "Values are filterable at search time."
        ),
    )


class IngestRequest(BaseModel):
    items: list[IngestItem] = Field(
        ...,
        min_length=1,
        description="One or more items to ingest. Batch size ≤ 100 recommended.",
    )


class IngestResult(BaseModel):
    ingested: int
    ids: list[str]
    embedding_model: str | None = None
    processor: ProcessorJobResult | None = None


class MemoryItem(BaseModel):
    """A stored memory item as returned by the API."""

    id: str
    dataset_key: str
    raw_text: str
    metadata: dict[str, Any]
    embedding_model: str | None
    created_at: str | None
    updated_at: str | None = None
    is_deleted: bool = False


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Raw text query. CortexDB embeds it and runs cosine similarity search.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return.",
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional exact-match filters on item metadata. "
            "All specified key-value pairs must match."
        ),
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0–1) to include in results.",
    )
    keyword_query: str | None = Field(
        default=None,
        description=(
            "Optional keyword substring to match against raw_text (case-insensitive). "
            "When provided alongside query, scores are blended: "
            "final = vector_weight * vector_score + (1 - vector_weight) * keyword_score."
        ),
    )
    vector_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight applied to the vector score in hybrid mode (0–1). "
            "1.0 = pure vector search (default). "
            "0.5 = equal blend. 0.0 = pure keyword search."
        ),
    )


class SearchHit(BaseModel):
    item: MemoryItem
    score: float = Field(description="Final blended score (0–1). Higher is more similar.")
    vector_score: float | None = Field(
        default=None,
        description="Raw cosine similarity component before blending.",
    )
    keyword_score: float | None = Field(
        default=None,
        description="Keyword match score component before blending (1.0 if keyword matched, 0.0 if not).",
    )


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    query: str
    embedding_model: str | None
    total_searched: int = Field(description="Number of items scored before top_k filter.")
    search_mode: str = Field(
        default="vector",
        description="Mode used: 'vector', 'keyword', or 'hybrid'.",
    )
