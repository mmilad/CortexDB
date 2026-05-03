"""stdio transport for the CortexDB MCP server.

Implements the MCP JSON-RPC 2.0 protocol over stdin/stdout, enabling
CortexDB to act as a local MCP server for agent frameworks that launch
MCP servers as subprocesses (Claude Desktop, Cursor, Continue, etc.).

Usage
-----
Run directly::

    python -m app.mcp.stdio

Or as the CLI entry point configured in pyproject.toml::

    cortexdb-mcp

The process reads newline-delimited JSON-RPC messages from stdin and
writes JSON-RPC responses to stdout.  Diagnostic output goes to stderr so
it does not corrupt the protocol stream.

All MCP methods supported by the HTTP endpoint are supported here:
  initialize, ping, notifications/initialized,
  resources/list, resources/read,
  tools/list, tools/call

Configuration
-------------
Uses the same environment variables as the HTTP server:

  CORTEXDB_DB_PATH          — path to the SQLite database file
  CORTEXDB_EMBED_PROVIDER   — embedding provider (default: ollama)
  CORTEXDB_EMBED_MODEL      — embedding model
  CORTEXDB_EMBED_URL        — embedding endpoint URL
  CORTEXDB_EMBED_API_KEY    — API key for 'api' provider

Embedding is not needed for MCP operations (registry read-only).
Set ``CORTEXDB_EMBED_PROVIDER=none`` to skip the Ollama startup delay
unless you intend to use ingest/search via the HTTP API at the same time.
"""

from __future__ import annotations

import json
import logging
import sys

# Redirect all logging to stderr — stdout is reserved for MCP protocol output.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("cortexdb.mcp.stdio")


def _ok(result: object, req_id: object = None) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(code: int, message: str, req_id: object = None) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Entry point: read JSON-RPC from stdin, dispatch, write to stdout."""
    # Import here so that environment variables set before invocation are picked up.
    from app.mcp.server import _HANDLERS
    from app.store import close_store, get_store

    store = get_store()
    logger.info("CortexDB MCP stdio server started. db=%s", store)

    try:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                body = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                _write(_err(-32700, f"Parse error: {exc}"))
                continue

            req_id = body.get("id")
            method = body.get("method", "")
            params = body.get("params") or {}
            is_notification = "id" not in body

            handler = _HANDLERS.get(method)
            if handler is None:
                if not is_notification:
                    _write(_err(-32601, f"Method not found: {method}", req_id))
                # Notifications with unknown methods are silently ignored per spec.
                continue

            try:
                result = handler(params, store)
                if not is_notification:
                    _write(_ok(result, req_id))
            except ValueError as exc:
                if not is_notification:
                    _write(_err(-32602, str(exc), req_id))
            except Exception as exc:
                logger.exception("Internal error handling method '%s'", method)
                if not is_notification:
                    _write(_err(-32603, f"Internal error: {exc}", req_id))
    finally:
        close_store()


if __name__ == "__main__":
    main()
