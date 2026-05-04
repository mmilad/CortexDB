"""Dynamic MCP (Model Context Protocol) server for CortexDB.

Exposes all registered datasets as MCP resources and all registered tools
as MCP tool descriptors. Resources and tools are generated dynamically from
the SQLite registry — adding a new dataset via POST /datasets immediately
makes it discoverable through MCP without any code changes.

MCP Protocol Reference: https://spec.modelcontextprotocol.io/

Transport
---------
This module implements the MCP JSON-RPC 2.0 message format over HTTP POST.
Mount it at /mcp in the FastAPI app. It accepts raw MCP JSON-RPC bodies and
returns JSON-RPC responses.

For stdio transport (CLI agents): use the standalone entry point app/mcp/stdio.py.

MCP capabilities exposed
------------------------
  resources/list          → one resource per dataset + cortexdb://context/index
                             + cortexdb://graph
  resources/read          → full context payload for a resource URI
  tools/list              → one tool per ToolRecord entry
  tools/call              → proxies to the REST API (passthrough, no internal logic)
  initialize              → protocol handshake
  ping                    → liveness

Resource URI scheme
-------------------
  cortexdb://datasets/{dataset_key}   — one resource per dataset
  cortexdb://tools/{tool_key}         — one resource per tool (browseable metadata)
  cortexdb://context/index            — minimal orientation index
  cortexdb://graph                    — full relationship map

Token budget notes
------------------
  resources/list descriptions are capped to ~100 tokens each.
  Full content is only returned on resources/read.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app import __version__
from app.context_builders import (
    build_context_index,
    build_dataset_payload,
    build_graph_payload,
    build_tool_payload,
)
from app.schemas.dataset import DatasetRecord
from app.schemas.tool import ToolRecord
from app.store import SqliteStore, get_store

logger = logging.getLogger("cortexdb.mcp")

# Cache for remotely fetched JSON schemas: url → (schema_dict, fetched_at)
_schema_cache: dict[str, tuple[dict, float]] = {}
_SCHEMA_CACHE_TTL = 300.0  # seconds

router = APIRouter(prefix="/mcp", tags=["mcp"])

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "cortexdb", "version": __version__}


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------


def _ok(result: Any, req_id: Any = None) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(code: int, message: str, req_id: Any = None) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _dataset_description(rec: DatasetRecord) -> str:
    summary = rec.llm_summary or rec.semantic_description
    caps = ", ".join(rec.retrieval_capabilities) if rec.retrieval_capabilities else "none"
    related = ", ".join(rec.relationship_hints[:3]) if rec.relationship_hints else "none"
    # Keep under ~100 tokens
    desc = f"{summary} | capabilities: {caps} | related: {related}"
    return desc[:500]


def _tool_description(rec: ToolRecord) -> str:
    summary = rec.llm_summary or rec.description
    scope = (
        (rec.safety_scope if isinstance(rec.safety_scope, str) else ", ".join(rec.safety_scope))
        if rec.safety_scope
        else "none"
    )
    desc = f"{summary} | safety_scope: {scope}"
    return desc[:400]


def _dataset_resource(rec: DatasetRecord) -> dict:
    return {
        "uri": f"cortexdb://datasets/{rec.dataset_key}",
        "name": rec.display_name,
        "description": _dataset_description(rec),
        "mimeType": "application/json",
    }


def _tool_metadata_resource(rec: ToolRecord) -> dict:
    return {
        "uri": f"cortexdb://tools/{rec.tool_key}",
        "name": rec.name,
        "description": _tool_description(rec),
        "mimeType": "application/json",
    }


def _dataset_content(rec: DatasetRecord) -> str:
    return json.dumps(build_dataset_payload(rec), indent=2)


def _tool_content(rec: ToolRecord) -> str:
    return json.dumps(build_tool_payload(rec), indent=2)


# ---------------------------------------------------------------------------
# MCP method handlers
# ---------------------------------------------------------------------------


def _handle_initialize(params: dict, store: SqliteStore) -> dict:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": SERVER_INFO,
        "capabilities": {
            "resources": {"subscribe": False, "listChanged": False},
            "tools": {},
        },
    }


def _handle_resources_list(params: dict, store: SqliteStore) -> dict:
    resources = []

    # Static resources
    resources.append({
        "uri": "cortexdb://context/index",
        "name": "CortexDB Context Index",
        "description": (
            "Minimal token-efficient orientation index of all datasets and tools. "
            "Call this first to discover what exists before fetching full context."
        ),
        "mimeType": "application/json",
    })
    resources.append({
        "uri": "cortexdb://graph",
        "name": "CortexDB Relationship Graph",
        "description": (
            "Full relationship map between datasets and tools. "
            "Use to understand data flow and plan multi-dataset queries."
        ),
        "mimeType": "application/json",
    })

    # Dynamic: one resource per dataset
    for data in store.list_datasets().values():
        rec = DatasetRecord(**data)
        if rec.status == "active":
            resources.append(_dataset_resource(rec))

    # Dynamic: one metadata resource per tool
    for data in store.list_tools().values():
        rec = ToolRecord(**data)
        if rec.status == "active":
            resources.append(_tool_metadata_resource(rec))

    return {"resources": resources}


def _handle_resources_read(params: dict, store: SqliteStore) -> dict:
    uri: str = params.get("uri", "")

    if uri == "cortexdb://context/index":
        index = build_context_index(store)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(index, indent=2),
                }
            ]
        }

    if uri == "cortexdb://graph":
        graph = build_graph_payload(store)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(graph, indent=2),
                }
            ]
        }

    if uri.startswith("cortexdb://datasets/"):
        key = uri[len("cortexdb://datasets/"):]
        data = store.get_dataset(key)
        if not data:
            raise ValueError(f"Dataset '{key}' not found")
        rec = DatasetRecord(**data)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": _dataset_content(rec),
                }
            ]
        }

    if uri.startswith("cortexdb://tools/"):
        key = uri[len("cortexdb://tools/"):]
        data = store.get_tool(key)
        if not data:
            raise ValueError(f"Tool '{key}' not found")
        rec = ToolRecord(**data)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": _tool_content(rec),
                }
            ]
        }

    raise ValueError(f"Unknown resource URI: {uri}")


def _resolve_input_schema(ref: str | None) -> dict[str, Any]:
    """Resolve an input_schema_ref to an inline JSON Schema dict.

    Resolution order:
      1. If *ref* is already valid JSON → parse and return.
      2. If *ref* starts with http:// or https:// → fetch with a 5 s timeout,
         cache result for 5 minutes, return on success.
      3. Fallback: return a minimal ``{"type": "object"}`` schema.
    """
    if not ref:
        return {"type": "object", "properties": {}}

    # Attempt inline JSON parse first
    try:
        parsed = json.loads(ref)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt URL fetch
    if ref.startswith("http://") or ref.startswith("https://"):
        now = time.monotonic()
        cached = _schema_cache.get(ref)
        if cached is not None:
            schema, fetched_at = cached
            if now - fetched_at < _SCHEMA_CACHE_TTL:
                return schema

        try:
            resp = httpx.get(ref, timeout=5.0, follow_redirects=True)
            resp.raise_for_status()
            schema = resp.json()
            if isinstance(schema, dict):
                _schema_cache[ref] = (schema, now)
                return schema
        except Exception as exc:
            logger.warning("Could not fetch input_schema_ref '%s': %s", ref, exc)

    return {"type": "object", "properties": {}}


def _handle_tools_list(params: dict, store: SqliteStore) -> dict:
    """Return MCP tool descriptors generated from ToolRecord entries."""
    tools = []
    for data in store.list_tools().values():
        rec = ToolRecord(**data)
        if rec.status != "active":
            continue
        tools.append(
            {
                "name": rec.tool_key,
                "description": _tool_description(rec),
                "inputSchema": _resolve_input_schema(rec.input_schema_ref),
            }
        )
    return {"tools": tools}


def _handle_tools_call(params: dict, store: SqliteStore) -> dict:
    """Passthrough stub — actual tool execution is external.

    CortexDB does not execute tools internally (no LLM inside).
    This returns the tool metadata so the caller can execute it externally.
    """
    name = params.get("name", "")
    data = store.get_tool(name)
    if not data:
        raise ValueError(f"Tool '{name}' not found in registry")
    rec = ToolRecord(**data)
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"CortexDB does not execute tools internally. "
                    f"Tool '{name}' is registered with: "
                    f"input_schema_ref={rec.input_schema_ref}, "
                    f"output_schema_ref={rec.output_schema_ref}. "
                    f"Execute externally using the tool's actual endpoint."
                ),
            }
        ],
        "isError": False,
    }


def _handle_ping(params: dict, store: SqliteStore) -> dict:
    return {}


def _handle_notifications_initialized(params: dict, store: SqliteStore) -> dict:
    """No-op acknowledgement for the MCP post-handshake notification."""
    return {}


_HANDLERS = {
    "initialize": _handle_initialize,
    "notifications/initialized": _handle_notifications_initialized,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": _handle_ping,
}


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


@router.post(
    "",
    summary="MCP JSON-RPC endpoint",
    description=(
        "Accepts MCP JSON-RPC 2.0 requests. "
        "Supported methods: initialize, ping, resources/list, resources/read, "
        "tools/list, tools/call. "
        "Resources and tools are generated dynamically from the registry — "
        "no restart needed when datasets or tools are added."
    ),
    response_class=JSONResponse,
)
async def mcp_endpoint(
    request: Request,
    store: SqliteStore = Depends(get_store),
) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_err(-32700, "Parse error"), status_code=400)

    req_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params") or {}

    # MCP notifications have no "id" — they are fire-and-forget; return 202.
    is_notification = "id" not in body

    handler = _HANDLERS.get(method)
    if handler is None:
        if is_notification:
            return Response(status_code=202)
        return JSONResponse(_err(-32601, f"Method not found: {method}", req_id))

    try:
        result = handler(params, store)
        if is_notification:
            return Response(status_code=202)
        return JSONResponse(_ok(result, req_id))
    except ValueError as exc:
        if is_notification:
            return Response(status_code=202)
        return JSONResponse(_err(-32602, str(exc), req_id))
    except Exception as exc:
        if is_notification:
            return Response(status_code=202)
        return JSONResponse(_err(-32603, f"Internal error: {exc}", req_id), status_code=500)
