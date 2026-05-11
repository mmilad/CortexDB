from typing import Any

from pydantic import BaseModel, Field


class ToolQueryExample(BaseModel):
    """Example invocation pattern for a tool."""

    label: str = Field(..., description="Short label for this invocation pattern.")
    description: str = Field(..., description="One sentence on when to use this pattern.")
    example_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Representative input payload matching input_schema_ref.",
    )


class ToolRecord(BaseModel):
    tool_key: str = Field(..., description="Stable tool identifier")
    name: str
    description: str

    # LLM-guidance fields
    llm_summary: str | None = Field(
        default=None,
        description=(
            "1–2 sentence plain-English summary for LLM context. "
            "Answers 'what does this tool do and when should I call it?'. "
            "Used by GET /context/index and MCP tool descriptions."
        ),
    )
    query_examples: list[ToolQueryExample] = Field(
        default_factory=list,
        description="Concrete invocation examples an LLM can adapt.",
    )

    capability_tags: list[str] = Field(default_factory=list)
    relationship_hints: list[str] = Field(
        default_factory=list,
        description=(
            "Informal relationship hints (legacy). Prefer POST /relationships "
            "for machine-traversable edges."
        ),
    )
    embedding_model_version: str | None = None
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None
    input_schema_ref: str | None = Field(
        default=None, description="JSON Schema pointer for tool inputs."
    )
    output_schema_ref: str | None = Field(
        default=None, description="JSON Schema pointer for tool outputs."
    )
    safety_scope: str | list[str] | None = Field(
        default=None,
        description="Human-readable constraints or scopes for safe use.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-defined extension; not interpreted by CortexDB.",
    )
