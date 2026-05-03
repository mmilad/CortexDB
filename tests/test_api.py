"""Integration tests for the FastAPI application.

Uses CORTEXDB_EMBED_PROVIDER=none to avoid requiring Ollama.
All embedding-dependent paths are tested separately in test_store.py.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Disable embedding for all tests in this module
os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"


@pytest.fixture(scope="module")
def client():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    os.environ["CORTEXDB_DB_PATH"] = db_path

    # Import after env vars are set so singletons pick them up
    from app.store import _store  # noqa: F401
    import app.store as store_mod
    import app.embed.service as embed_mod

    # Reset singletons so each test module gets fresh state
    store_mod._store = None
    embed_mod._service = embed_mod.EmbeddingService()

    from app.main import app
    with TestClient(app) as c:
        yield c

    store_mod.close_store()
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASET_PAYLOAD = {
    "dataset_key": "test_ds",
    "display_name": "Test Dataset",
    "schema_version": "v1",
    "semantic_description": "A test dataset for unit tests",
    "usage_guidance": "Use in tests only",
    "retrieval_capabilities": ["filter_only"],
    "llm_summary": "Test dataset for testing",
}


def test_create_dataset(client):
    r = client.post("/datasets", json=DATASET_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["dataset_key"] == "test_ds"


def test_list_datasets(client):
    r = client.get("/datasets")
    assert r.status_code == 200
    keys = [d["dataset_key"] for d in r.json()]
    assert "test_ds" in keys


def test_get_dataset(client):
    r = client.get("/datasets/test_ds")
    assert r.status_code == 200
    assert r.json()["display_name"] == "Test Dataset"


def test_get_dataset_not_found(client):
    r = client.get("/datasets/does_not_exist")
    assert r.status_code == 404


def test_validate_dataset(client):
    r = client.post("/datasets/test_ds/validate")
    assert r.status_code == 200
    data = r.json()
    assert data["last_validated_at"] is not None


def test_delete_dataset(client):
    r = client.post("/datasets", json={**DATASET_PAYLOAD, "dataset_key": "to_delete"})
    assert r.status_code == 200
    r2 = client.delete("/datasets/to_delete")
    assert r2.status_code == 200
    r3 = client.get("/datasets/to_delete")
    assert r3.status_code == 404


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOL_PAYLOAD = {
    "tool_key": "test_tool",
    "name": "Test Tool",
    "description": "A tool for testing",
    "llm_summary": "Use this tool in tests",
    "status": "active",
}


def test_create_tool(client):
    r = client.post("/tools", json=TOOL_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["tool_key"] == "test_tool"


def test_list_tools(client):
    r = client.get("/tools")
    assert r.status_code == 200
    keys = [t["tool_key"] for t in r.json()]
    assert "test_tool" in keys


def test_get_tool(client):
    r = client.get("/tools/test_tool")
    assert r.status_code == 200


def test_get_tool_not_found(client):
    r = client.get("/tools/nope")
    assert r.status_code == 404


def test_delete_tool(client):
    r = client.post("/tools", json={**TOOL_PAYLOAD, "tool_key": "del_tool"})
    assert r.status_code == 200
    r2 = client.delete("/tools/del_tool")
    assert r2.status_code == 200
    r3 = client.get("/tools/del_tool")
    assert r3.status_code == 404


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

REL_PAYLOAD = {
    "source_type": "dataset",
    "source_key": "test_ds",
    "target_type": "tool",
    "target_key": "test_tool",
    "edge_type": "consumes",
    "description": "Tool reads from test dataset",
}


def test_create_relationship_auto_id(client):
    r = client.post("/relationships", json=REL_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["id"] != ""
    assert data["edge_type"] == "consumes"


def test_list_relationships(client):
    r = client.get("/relationships")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_list_relationships_filter(client):
    r = client.get("/relationships?node_key=test_ds")
    assert r.status_code == 200
    for rel in r.json():
        assert rel["source_key"] == "test_ds" or rel["target_key"] == "test_ds"


def test_get_relationship(client):
    r = client.post("/relationships", json={**REL_PAYLOAD, "id": "fixed_id"})
    assert r.status_code == 200
    r2 = client.get("/relationships/fixed_id")
    assert r2.status_code == 200
    assert r2.json()["id"] == "fixed_id"


def test_delete_relationship(client):
    r = client.post("/relationships", json={**REL_PAYLOAD, "id": "to_del"})
    assert r.status_code == 200
    r2 = client.delete("/relationships/to_del")
    assert r2.status_code == 200
    r3 = client.get("/relationships/to_del")
    assert r3.status_code == 404


# ---------------------------------------------------------------------------
# Context endpoints
# ---------------------------------------------------------------------------

def test_context_index(client):
    r = client.get("/context/index")
    assert r.status_code == 200
    body = r.json()
    assert "datasets" in body
    assert "tools" in body
    assert "relationship_count" in body


def test_context_dataset(client):
    r = client.get("/context/dataset/test_ds")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "test_ds"
    assert "query_examples" in body


def test_context_dataset_not_found(client):
    r = client.get("/context/dataset/missing")
    assert r.status_code == 404


def test_context_tool(client):
    r = client.get("/context/tool/test_tool")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "test_tool"


def test_context_graph(client):
    r = client.get("/context/graph")
    assert r.status_code == 200
    assert "edges" in r.json()


# ---------------------------------------------------------------------------
# Graph explore
# ---------------------------------------------------------------------------

def test_graph_explore(client):
    r = client.get("/graph/explore?start=dataset:test_ds&depth=1")
    assert r.status_code == 200
    body = r.json()
    assert body["root"] == "test_ds"
    assert isinstance(body["nodes"], list)


# ---------------------------------------------------------------------------
# Memory items
# ---------------------------------------------------------------------------

def test_ingest_requires_embedding(client):
    r = client.post("/datasets/test_ds/ingest", json={"items": [{"raw_text": "hello"}]})
    assert r.status_code == 503


def test_search_keyword_only(client):
    """Keyword-only search works even when embedding is disabled."""
    # Insert a raw item manually via the store so we can test search without embed
    from app.store import get_store
    store = get_store()
    store.insert_memory_item({
        "id": "kw_test_1",
        "dataset_key": "test_ds",
        "raw_text": "unique keyword abcxyz",
        "metadata": {},
    })
    r = client.post(
        "/datasets/test_ds/search",
        json={"query": "ignored", "keyword_query": "abcxyz", "vector_weight": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["search_mode"] == "keyword"
    assert len(body["hits"]) == 1
    assert body["hits"][0]["item"]["raw_text"] == "unique keyword abcxyz"


def test_list_items(client):
    r = client.get("/datasets/test_ds/items")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_soft_delete_and_restore_via_list(client):
    from app.store import get_store
    store = get_store()
    store.insert_memory_item({
        "id": "soft_del_test",
        "dataset_key": "test_ds",
        "raw_text": "I will be soft deleted",
        "metadata": {},
    })

    r = client.delete("/datasets/test_ds/items/soft_del_test")
    assert r.status_code == 200
    assert r.json()["soft_deleted"] == "soft_del_test"

    r2 = client.get("/datasets/test_ds/items")
    ids = [i["id"] for i in r2.json()]
    assert "soft_del_test" not in ids

    r3 = client.get("/datasets/test_ds/items?include_deleted=true")
    ids_with = [i["id"] for i in r3.json()]
    assert "soft_del_test" in ids_with


def test_hard_delete(client):
    from app.store import get_store
    store = get_store()
    store.insert_memory_item({
        "id": "hard_del_test",
        "dataset_key": "test_ds",
        "raw_text": "gone forever",
        "metadata": {},
    })

    r = client.delete("/datasets/test_ds/items/hard_del_test/hard")
    assert r.status_code == 200
    assert r.json()["hard_deleted"] == "hard_del_test"

    r2 = client.get("/datasets/test_ds/items?include_deleted=true")
    ids = [i["id"] for i in r2.json()]
    assert "hard_del_test" not in ids


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_capabilities(client):
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["llm_inside"] is False
    assert "llm_context_endpoints" in body
    assert "mcp_endpoint" in body


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def _mcp(client, method, params=None, req_id=1):
    return client.post("/mcp", json={
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    })


def test_mcp_initialize(client):
    r = _mcp(client, "initialize")
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["protocolVersion"] == "2024-11-05"
    assert body["result"]["serverInfo"]["version"] == "0.3.0"


def test_mcp_ping(client):
    r = _mcp(client, "ping")
    assert r.status_code == 200
    assert r.json()["result"] == {}


def test_mcp_resources_list(client):
    r = _mcp(client, "resources/list")
    assert r.status_code == 200
    resources = r.json()["result"]["resources"]
    uris = [res["uri"] for res in resources]
    assert "cortexdb://context/index" in uris
    assert "cortexdb://graph" in uris
    assert any(u.startswith("cortexdb://datasets/") for u in uris)


def test_mcp_resources_read_index(client):
    r = _mcp(client, "resources/read", {"uri": "cortexdb://context/index"})
    assert r.status_code == 200
    content = r.json()["result"]["contents"][0]
    assert content["mimeType"] == "application/json"


def test_mcp_resources_read_dataset(client):
    r = _mcp(client, "resources/read", {"uri": "cortexdb://datasets/test_ds"})
    assert r.status_code == 200


def test_mcp_tools_list(client):
    r = _mcp(client, "tools/list")
    assert r.status_code == 200
    assert "tools" in r.json()["result"]


def test_mcp_notification_initialized(client):
    """notifications/initialized has no 'id' — should return 202, not an error."""
    r = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    })
    assert r.status_code == 202


def test_mcp_unknown_method_returns_error(client):
    r = _mcp(client, "nonexistent/method")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32601


def test_mcp_unknown_notification_returns_202(client):
    """Unknown notifications should be silently accepted per MCP spec."""
    r = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "unknown/notification",
        "params": {},
    })
    assert r.status_code == 202


# ---------------------------------------------------------------------------
# Dataset discover
# ---------------------------------------------------------------------------

def test_discover_no_match_suggests_create(client):
    r = client.post("/datasets/discover", json={"intent": "store quantum entanglement data"})
    assert r.status_code == 200
    body = r.json()
    assert "recommended_action" in body
