"""Typed, machine-traversable relationship between datasets and tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NodeType = Literal["dataset", "tool"]
EdgeType = Literal[
    "joins_on",       # datasets joined on shared fields
    "feeds_into",     # one dataset/tool produces data consumed by another
    "shared_entity",  # both nodes reference the same real-world entity type
    "produces",       # a tool produces records in a dataset
    "consumes",       # a tool reads from a dataset
    "related",        # generic semantic relationship
]


class RelationshipRecord(BaseModel):
    id: str = Field(
        default="",
        description="Stable relationship identifier. Auto-generated UUID when omitted.",
    )
    source_type: NodeType = Field(..., description="Type of the source node.")
    source_key: str = Field(..., description="dataset_key or tool_key of the source node.")
    target_type: NodeType = Field(..., description="Type of the target node.")
    target_key: str = Field(..., description="dataset_key or tool_key of the target node.")
    edge_type: EdgeType = Field(
        default="related",
        description=(
            "Semantic type of the relationship. Determines how traversal and "
            "LLM context summaries describe this edge."
        ),
    )
    join_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Field names used to join/correlate source and target "
            "(e.g. ['issue_id']). Only meaningful for joins_on / shared_entity."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "One plain-English sentence describing the relationship. "
            "Included verbatim in GET /context/graph and MCP resource descriptions."
        ),
    )
    created_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp set automatically by the store.",
    )


class GraphNode(BaseModel):
    key: str
    node_type: NodeType
    display_name: str | None = None
    llm_summary: str | None = None
    entity_types: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: EdgeType
    description: str = ""
    join_fields: list[str] = Field(default_factory=list)


class GraphExploreResponse(BaseModel):
    root: str
    depth: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
