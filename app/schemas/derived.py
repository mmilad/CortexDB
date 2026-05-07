"""Schemas for LLM-derived memory extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DERIVED_MEMORY_SCHEMA_VERSION = "cortexdb.derived_memory.v1"


class DerivedDatasetHint(BaseModel):
    display_name: str | None = None
    semantic_description: str | None = None
    usage_guidance: str | None = None
    entity_types: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)


class DerivedMemoryRecord(BaseModel):
    dataset_key: str = Field(
        ...,
        description="Target dataset key, preferably lowercase snake_case starting with derived_.",
    )
    kind: str = Field(
        default="custom",
        description="Memory kind such as fact, decision, goal, knowledge, preference, constraint, task, or event.",
    )
    text: str = Field(..., description="One durable memory item. No chat filler.")
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0 to 1.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataset: DerivedDatasetHint | None = None


class DerivedMemoryEnvelope(BaseModel):
    schema_version: str = DERIVED_MEMORY_SCHEMA_VERSION
    memories: list[DerivedMemoryRecord] = Field(default_factory=list)
