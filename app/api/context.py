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

from app.context_builders import (
    build_context_index,
    build_dataset_payload,
    build_graph_payload,
    build_tool_payload,
)
from app.schemas.dataset import DatasetRecord
from app.schemas.session import (
    ContextItem,
    ContextPackage,
    ContextRequest,
    SessionMessageRecord,
    SessionRecord,
    SessionSummaryRecord,
)
from app.schemas.tool import ToolRecord
from app.services.session import ensure_session
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
    payload = build_context_index(store)
    return ContextIndex(**payload)


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
    payload = build_dataset_payload(rec)
    return DatasetContext(
        key=rec.dataset_key,
        capabilities=payload["retrieval_capabilities"],
        retrieval_profiles=[rp.model_dump() for rp in rec.retrieval_profiles],
        schema_version=rec.schema_version,
        **{k: v for k, v in payload.items() if k not in ("dataset_key", "retrieval_capabilities")},
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
    payload = build_tool_payload(rec)
    return ToolContext(key=rec.tool_key, **{k: v for k, v in payload.items() if k != "tool_key"})


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
    payload = build_graph_payload(store)
    return GraphContext(**payload)


@router.post(
    "",
    response_model=ContextPackage,
    summary="Build a session-aware context package",
    description=(
        "Returns prompt-ready context from session history, session summaries, "
        "and relevant dataset memory. Disabled session chunks are represented "
        "through summaries instead of detailed messages."
    ),
)
def build_context_package(
    body: ContextRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> ContextPackage:
    session = ensure_session(store, session_id=body.session_id)
    summaries = [
        SessionSummaryRecord(**row)
        for row in store.list_session_summaries(session.id)
    ]
    messages = [
        SessionMessageRecord(**row)
        for row in store.list_session_messages(
            session.id,
            limit=body.top_k * 2,
            autocontext_only=True,
        )
    ]

    dataset_keys = body.dataset_keys or list(store.list_datasets().keys())
    items: list[ContextItem] = []
    for dataset_key in dataset_keys:
        if not store.get_dataset(dataset_key):
            continue
        for row in store.search_memory_items(
            dataset_key=dataset_key,
            query_vector=None,
            top_k=body.top_k,
            keyword_query=body.prompt,
            vector_weight=0.0,
        ):
            items.append(
                ContextItem(
                    kind="memory_item",
                    text=row["raw_text"],
                    source_id=row["id"],
                    dataset_key=dataset_key,
                    score=row.get("score"),
                    metadata=row.get("metadata", {}),
                )
            )

    items.sort(key=lambda item: item.score or 0.0, reverse=True)
    return ContextPackage(
        session=SessionRecord(**session.model_dump()),
        prompt=body.prompt,
        summaries=summaries,
        messages=messages,
        items=items[: body.top_k],
        dataset_keys=dataset_keys,
        usage_hint=(
            "Use summaries for compacted history, messages for recent chat turns, "
            "and items for dataset-backed memory. Final reasoning remains outside CortexDB."
        ),
    )
