"""LLM-optimised context endpoints for CortexDB.

These endpoints are designed specifically for low-token LLM consumption:

  GET /context/index          — Minimal orientation dump (~50 tokens/item).
                                Call this first to discover what exists.
  GET /context/dataset/{key}  — Full structured context for one dataset (~300 tokens).
  GET /context/tool/{key}     — Full structured context for one tool (~200 tokens).
  GET /context/graph          — Compact relationship map (~60 tokens/edge).

Workflow for an LLM agent:
  1. Call GET /context/index to orient.
  2. Identify relevant datasets/tools by key.
  3. Call GET /context/dataset/{key} for each relevant item.
  4. Optionally call GET /context/graph to understand data relationships.
  5. Query datasets using filterable_fields and query_examples.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.schemas.dataset import DatasetRecord
from app.schemas.tool import ToolRecord
from app.store import SqliteStore, get_store

router = APIRouter(prefix="/context", tags=["context"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DatasetIndexEntry(BaseModel):
    key: str
    display_name: str
    llm_summary: str | None
    capabilities: list[str]
    entity_types: list[str]
    access_patterns: list[str]
    status: str


class ToolIndexEntry(BaseModel):
    key: str
    name: str
    llm_summary: str | None
    capability_tags: list[str]
    status: str


class ContextIndex(BaseModel):
    """Minimal token-efficient index of all registered datasets and tools.

    LLM guidance: Load this first. Use the 'key' values to fetch full context
    with GET /context/dataset/{key} or GET /context/tool/{key}.
    """

    datasets: list[DatasetIndexEntry]
    tools: list[ToolIndexEntry]
    relationship_count: int = Field(
        description="Total number of registered relationships. "
        "Call GET /context/graph for the full map."
    )
    usage_hint: str = Field(
        default=(
            "Call GET /context/dataset/{key} for full query guidance on a specific dataset. "
            "Call GET /context/graph for relationship map. "
            "Call GET /relationships?node_key={key} for edges touching a node."
        )
    )


class DatasetContext(BaseModel):
    """Full LLM context for one dataset.

    Includes query examples, field descriptions, and relationship hints.
    LLM guidance: Use query_examples as templates. Adapt example_request payloads.
    """

    key: str
    display_name: str
    llm_summary: str | None
    semantic_description: str
    usage_guidance: str
    capabilities: list[str]
    content_kind: str
    entity_types: list[str]
    access_patterns: list[str]
    filterable_fields: list[str]
    field_descriptions: list[dict[str, Any]]
    query_examples: list[dict[str, Any]]
    relationship_hints: list[str]
    retrieval_profiles: list[dict[str, Any]]
    status: str
    schema_version: str


class ToolContext(BaseModel):
    """Full LLM context for one tool.

    LLM guidance: Use query_examples as invocation templates.
    Check safety_scope before calling.
    """

    key: str
    name: str
    llm_summary: str | None
    description: str
    capability_tags: list[str]
    safety_scope: Any
    input_schema_ref: str | None
    output_schema_ref: str | None
    query_examples: list[dict[str, Any]]
    relationship_hints: list[str]
    status: str


class GraphContextEdge(BaseModel):
    from_key: str
    from_type: str
    to_key: str
    to_type: str
    edge_type: str
    description: str
    join_fields: list[str]


class GraphContext(BaseModel):
    """Compact relationship map for LLM navigation.

    LLM guidance: Use this to understand which datasets/tools share data or
    can be joined. Follow edges to determine query sequencing.
    """

    edges: list[GraphContextEdge]
    usage_hint: str = Field(
        default=(
            "Traverse edges to plan multi-dataset queries. "
            "Use GET /graph/explore?start=dataset:{key}&depth=2 for BFS subgraph."
        )
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/index",
    response_model=ContextIndex,
    summary="Minimal token-efficient orientation index",
    description=(
        "Returns a compact summary of all registered datasets and tools. "
        "Designed as a low-token first call for LLM agents to orient before "
        "fetching full context for specific items. "
        "Each entry is ~50 tokens; a registry with 20 items is ~1000 tokens total."
    ),
)
def get_context_index(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> ContextIndex:
    datasets = store.list_datasets()
    tools = store.list_tools()
    rel_count = len(store.adjacency())

    ds_entries = []
    for d in datasets.values():
        rec = DatasetRecord(**d)
        ds_entries.append(
            DatasetIndexEntry(
                key=rec.dataset_key,
                display_name=rec.display_name,
                llm_summary=rec.llm_summary,
                capabilities=list(rec.retrieval_capabilities),
                entity_types=rec.entity_types,
                access_patterns=rec.access_patterns,
                status=rec.status,
            )
        )

    tool_entries = []
    for t in tools.values():
        rec = ToolRecord(**t)
        tool_entries.append(
            ToolIndexEntry(
                key=rec.tool_key,
                name=rec.name,
                llm_summary=rec.llm_summary,
                capability_tags=rec.capability_tags,
                status=rec.status,
            )
        )

    return ContextIndex(
        datasets=ds_entries,
        tools=tool_entries,
        relationship_count=rel_count,
    )


@router.get(
    "/dataset/{dataset_key}",
    response_model=DatasetContext,
    summary="Full LLM context for one dataset",
    description=(
        "Returns complete query guidance for a specific dataset including "
        "filterable fields, field glossary, query examples, and access patterns. "
        "~200–400 tokens per call. Fetch only the datasets you need."
    ),
)
def get_dataset_context(
    dataset_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> DatasetContext:
    data = store.get_dataset(dataset_key)
    if not data:
        raise HTTPException(status_code=404, detail="dataset not found")
    rec = DatasetRecord(**data)
    return DatasetContext(
        key=rec.dataset_key,
        display_name=rec.display_name,
        llm_summary=rec.llm_summary,
        semantic_description=rec.semantic_description,
        usage_guidance=rec.usage_guidance,
        capabilities=list(rec.retrieval_capabilities),
        content_kind=rec.content_kind,
        entity_types=rec.entity_types,
        access_patterns=rec.access_patterns,
        filterable_fields=rec.filterable_fields,
        field_descriptions=[fd.model_dump() for fd in rec.field_descriptions],
        query_examples=[qe.model_dump() for qe in rec.query_examples],
        relationship_hints=rec.relationship_hints,
        retrieval_profiles=[rp.model_dump() for rp in rec.retrieval_profiles],
        status=rec.status,
        schema_version=rec.schema_version,
    )


@router.get(
    "/tool/{tool_key}",
    response_model=ToolContext,
    summary="Full LLM context for one tool",
    description=(
        "Returns complete invocation guidance for a specific tool including "
        "input/output schema references, query examples, and safety scope. "
        "~150–300 tokens per call."
    ),
)
def get_tool_context(
    tool_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> ToolContext:
    data = store.get_tool(tool_key)
    if not data:
        raise HTTPException(status_code=404, detail="tool not found")
    rec = ToolRecord(**data)
    return ToolContext(
        key=rec.tool_key,
        name=rec.name,
        llm_summary=rec.llm_summary,
        description=rec.description,
        capability_tags=rec.capability_tags,
        safety_scope=rec.safety_scope,
        input_schema_ref=rec.input_schema_ref,
        output_schema_ref=rec.output_schema_ref,
        query_examples=[qe.model_dump() for qe in rec.query_examples],
        relationship_hints=rec.relationship_hints,
        status=rec.status,
    )


@router.get(
    "/graph",
    response_model=GraphContext,
    summary="Compact relationship map for LLM navigation",
    description=(
        "Returns all registered relationships in a compact form suitable for "
        "LLM reasoning about data flow and query sequencing. "
        "~60 tokens per edge. For BFS subgraph traversal use GET /graph/explore."
    ),
)
def get_context_graph(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> GraphContext:
    edges = store.adjacency()
    result = []
    for e in edges:
        result.append(
            GraphContextEdge(
                from_key=e["source_key"],
                from_type=e["source_type"],
                to_key=e["target_key"],
                to_type=e["target_type"],
                edge_type=e["edge_type"],
                description=e.get("description", ""),
                join_fields=e.get("join_fields", []),
            )
        )
    return GraphContext(edges=result)
