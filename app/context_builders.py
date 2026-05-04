"""Shared builders for LLM-oriented context payloads.

Used by both the REST context endpoints (app/api/context.py) and the MCP
resource handlers (app/mcp/server.py) to ensure a single source of truth
for the shape of each payload.
"""

from __future__ import annotations

from typing import Any

from app.schemas.dataset import DatasetRecord
from app.schemas.tool import ToolRecord
from app.store import SqliteStore


def build_context_index(store: SqliteStore) -> dict[str, Any]:
    """Return the minimal orientation index consumed by LLM agents."""
    datasets = store.list_datasets()
    tools = store.list_tools()
    rel_count = len(store.adjacency())

    return {
        "datasets": [
            {
                "key": (rec := DatasetRecord(**d)).dataset_key,
                "display_name": rec.display_name,
                "llm_summary": rec.llm_summary,
                "capabilities": list(rec.retrieval_capabilities),
                "entity_types": rec.entity_types,
                "access_patterns": rec.access_patterns,
                "status": rec.status,
            }
            for d in datasets.values()
        ],
        "tools": [
            {
                "key": (rec := ToolRecord(**t)).tool_key,
                "name": rec.name,
                "llm_summary": rec.llm_summary,
                "capability_tags": rec.capability_tags,
                "status": rec.status,
            }
            for t in tools.values()
        ],
        "relationship_count": rel_count,
        "usage_hint": (
            "Call GET /context/dataset/{key} for full query guidance on a specific dataset. "
            "Call GET /context/graph for relationship map. "
            "Call GET /relationships?node_key={key} for edges touching a node."
        ),
    }


def build_graph_payload(store: SqliteStore) -> dict[str, Any]:
    """Return the compact relationship map consumed by LLM agents."""
    edges = store.adjacency()
    return {
        "edges": [
            {
                "from_key": e["source_key"],
                "from_type": e["source_type"],
                "to_key": e["target_key"],
                "to_type": e["target_type"],
                "edge_type": e["edge_type"],
                "description": e.get("description", ""),
                "join_fields": e.get("join_fields", []),
            }
            for e in edges
        ],
        "usage_hint": (
            "Traverse edges to plan multi-dataset queries. "
            "Use GET /graph/explore?start=dataset:{key}&depth=2 for BFS subgraph."
        ),
    }


def build_dataset_payload(rec: DatasetRecord) -> dict[str, Any]:
    """Return the full context payload for a single dataset."""
    return {
        "dataset_key": rec.dataset_key,
        "display_name": rec.display_name,
        "llm_summary": rec.llm_summary,
        "semantic_description": rec.semantic_description,
        "usage_guidance": rec.usage_guidance,
        "retrieval_capabilities": list(rec.retrieval_capabilities),
        "content_kind": rec.content_kind,
        "entity_types": rec.entity_types,
        "access_patterns": rec.access_patterns,
        "filterable_fields": rec.filterable_fields,
        "field_descriptions": [fd.model_dump() for fd in rec.field_descriptions],
        "query_examples": [qe.model_dump() for qe in rec.query_examples],
        "relationship_hints": rec.relationship_hints,
        "status": rec.status,
    }


def build_tool_payload(rec: ToolRecord) -> dict[str, Any]:
    """Return the full context payload for a single tool."""
    return {
        "tool_key": rec.tool_key,
        "name": rec.name,
        "llm_summary": rec.llm_summary,
        "description": rec.description,
        "capability_tags": rec.capability_tags,
        "safety_scope": rec.safety_scope,
        "input_schema_ref": rec.input_schema_ref,
        "output_schema_ref": rec.output_schema_ref,
        "query_examples": [qe.model_dump() for qe in rec.query_examples],
        "relationship_hints": rec.relationship_hints,
        "status": rec.status,
    }
