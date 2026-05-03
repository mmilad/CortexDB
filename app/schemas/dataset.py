from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalCapability = Literal["vector", "keyword", "filter_only"]
ContentKind = Literal["documents", "events", "custom"]


class RetrievalProfile(BaseModel):
    """Named scoring or retrieval preset (deterministic; no internal LLM)."""

    name: str
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class QueryExample(BaseModel):
    """A concrete example query an LLM can adapt when calling this dataset."""

    label: str = Field(
        ...,
        description="Short human-readable label for this pattern (e.g. 'by_severity').",
    )
    description: str = Field(
        ...,
        description="One sentence explaining when to use this pattern.",
    )
    example_request: dict[str, Any] = Field(
        default_factory=dict,
        description="Representative filter/query payload; keys match filterable_fields.",
    )


class FieldDescription(BaseModel):
    """Per-field glossary entry so an LLM knows what a filterable field means."""

    field: str
    description: str
    example_values: list[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    dataset_key: str = Field(..., description="Stable dataset identifier")
    display_name: str
    schema_version: str
    semantic_description: str
    usage_guidance: str

    # LLM-guidance fields — optional but strongly recommended for MCP/context endpoints.
    llm_summary: str | None = Field(
        default=None,
        description=(
            "1–2 sentence plain-English summary optimised for LLM context. "
            "Answers 'what is this dataset and when should I query it?'. "
            "Used by GET /context/index and MCP resource descriptions."
        ),
    )
    query_examples: list[QueryExample] = Field(
        default_factory=list,
        description=(
            "Concrete query patterns an LLM can adapt. Each entry has a label, "
            "description, and an example_request payload."
        ),
    )
    field_descriptions: list[FieldDescription] = Field(
        default_factory=list,
        description="Glossary for filterable_fields so an LLM knows valid values and meanings.",
    )
    access_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Named patterns this dataset is designed for "
            "(e.g. 'by_time_range', 'by_entity_id', 'semantic_search')."
        ),
    )
    entity_types: list[str] = Field(
        default_factory=list,
        description=(
            "Real-world entity types stored here (e.g. 'Issue', 'Document', 'Event'). "
            "Used by the graph traversal service for typed node descriptions."
        ),
    )

    relationship_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Informal relationship hints (legacy). Prefer POST /relationships for "
            "machine-traversable edges."
        ),
    )
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
        description="Caller-defined extension; not interpreted by CortexDB.",
    )
