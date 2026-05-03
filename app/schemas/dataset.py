from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalCapability = Literal["vector", "keyword", "filter_only"]
ContentKind = Literal["documents", "events", "custom"]


class RetrievalProfile(BaseModel):
    """Named scoring or retrieval preset (deterministic; no internal LLM)."""

    name: str
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class DatasetRecord(BaseModel):
    dataset_key: str = Field(..., description="Stable dataset identifier")
    display_name: str
    schema_version: str
    semantic_description: str
    usage_guidance: str
    relationship_hints: list[str] = Field(default_factory=list)
    filterable_fields: list[str] = Field(default_factory=list)
    status: str = "active"
    content_kind: ContentKind = Field(
        default="custom",
        description="Semantic category (e.g. documents); not the physical row schema.",
    )
    retrieval_capabilities: list[RetrievalCapability] = Field(
        default_factory=list,
        description="What this dataset supports: vector, keyword, filter_only.",
    )
    capability_tags: list[str] = Field(
        default_factory=list,
        description="Tags for discovery and UI; caller-defined.",
    )
    table_refs: list[str] = Field(
        default_factory=list,
        description="Physical table or view names when backed by a store.",
    )
    retrieval_profiles: list[RetrievalProfile] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-defined extension; not interpreted by CortexDB v1.",
    )
