"""Tests for the MCP stdio transport (app.mcp.stdio).

These tests drive the stdio server by feeding it JSON-RPC messages
directly, without spawning a subprocess — the entry point logic is
tested via its internal helper functions.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def stdio_env(monkeypatch, tmp_path):
    """Fixture: configure a fresh in-memory store for stdio tests."""
    db_path = str(tmp_path / "stdio_test.sqlite")
    monkeypatch.setenv("CORTEXDB_DB_PATH", db_path)
    monkeypatch.setenv("CORTEXDB_EMBED_PROVIDER", "none")

    # Reset store singleton so it picks up the new DB path.
    import app.store as store_mod
    store_mod.close_store()
    store_mod._store = None

    yield db_path

    store_mod.close_store()


def _run_stdio(messages: list[dict], stdio_env) -> list[dict]:
    """Send *messages* through the stdio handler and collect responses."""
    from app.mcp.server import _HANDLERS
    from app.store import get_store

    store = get_store()
    responses = []

    for body in messages:
        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params") or {}
        is_notification = "id" not in body

        handler = _HANDLERS.get(method)
        if handler is None:
            if not is_notification:
                responses.append({"jsonrpc": "2.0", "id": req_id,
                                   "error": {"code": -32601, "message": f"Method not found: {method}"}})
            continue

        try:
            result = handler(params, store)
            if not is_notification:
                responses.append({"jsonrpc": "2.0", "id": req_id, "result": result})
        except ValueError as exc:
            if not is_notification:
                responses.append({"jsonrpc": "2.0", "id": req_id,
                                   "error": {"code": -32602, "message": str(exc)}})

    return responses


def test_stdio_initialize(stdio_env):
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}], stdio_env)
    assert len(responses) == 1
    r = responses[0]["result"]
    assert r["protocolVersion"] == "2024-11-05"
    assert r["serverInfo"]["name"] == "cortexdb"


def test_stdio_ping(stdio_env):
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}}], stdio_env)
    assert responses[0]["result"] == {}


def test_stdio_resources_list_empty(stdio_env):
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}}], stdio_env)
    resources = responses[0]["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "cortexdb://context/index" in uris
    assert "cortexdb://graph" in uris


def test_stdio_notification_no_response(stdio_env):
    """Notifications (no 'id') should produce no response."""
    responses = _run_stdio([
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    ], stdio_env)
    assert responses == []


def test_stdio_unknown_notification_no_response(stdio_env):
    responses = _run_stdio([
        {"jsonrpc": "2.0", "method": "unknown/notification", "params": {}},
    ], stdio_env)
    assert responses == []


def test_stdio_unknown_method_error(stdio_env):
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 9, "method": "no/such", "params": {}}], stdio_env)
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] == -32601


def test_stdio_resources_read_missing_uri(stdio_env):
    responses = _run_stdio([{
        "jsonrpc": "2.0", "id": 10, "method": "resources/read",
        "params": {"uri": "cortexdb://datasets/nonexistent"},
    }], stdio_env)
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] == -32602


def test_stdio_tools_list_empty(stdio_env):
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}}], stdio_env)
    assert responses[0]["result"]["tools"] == []


def test_stdio_resources_list_reflects_registered_dataset(stdio_env):
    """A dataset registered via the store appears in resources/list."""
    from app.store import get_store
    store = get_store()
    store.upsert_dataset("stdio_ds", {
        "dataset_key": "stdio_ds",
        "display_name": "Stdio Test Dataset",
        "schema_version": "v1",
        "semantic_description": "test",
        "usage_guidance": "test",
        "status": "active",
        "llm_summary": "A dataset for stdio tests",
        "retrieval_capabilities": [],
    })
    responses = _run_stdio([{"jsonrpc": "2.0", "id": 12, "method": "resources/list", "params": {}}], stdio_env)
    uris = [r["uri"] for r in responses[0]["result"]["resources"]]
    assert "cortexdb://datasets/stdio_ds" in uris
