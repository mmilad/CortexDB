from typing import Any

from pydantic import BaseModel, Field


class ToolRecord(BaseModel):
    tool_key: str = Field(..., description="Stable tool identifier")
    name: str
    description: str
    capability_tags: list[str] = Field(default_factory=list)
    relationship_hints: list[str] = Field(default_factory=list)
    embedding_model_version: str | None = None
    status: str = "active"
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
        description="Caller-defined extension; not interpreted by CortexDB v1.",
    )
