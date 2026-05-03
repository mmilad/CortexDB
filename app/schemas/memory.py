"""Schemas for memory items — the unit of storable + searchable content in CortexDB."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


class MemoryItem(BaseModel):
    """A stored memory item as returned by the API."""

    id: str
    dataset_key: str
    raw_text: str
    metadata: dict[str, Any]
    embedding_model: str | None
    created_at: str | None


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


class SearchHit(BaseModel):
    item: MemoryItem
    score: float = Field(description="Cosine similarity score (0–1). Higher is more similar.")


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    query: str
    embedding_model: str | None
    total_searched: int = Field(description="Number of items scored before top_k filter.")
