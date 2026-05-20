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
    import app.store.main as store_mod
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


def test_delete_dataset_cascades_items(client):
    """Deleting a dataset removes its memory items from the store."""
    r = client.post("/datasets", json={**DATASET_PAYLOAD, "dataset_key": "ds_with_items"})
    assert r.status_code == 200
    from app.store import get_store
    store = get_store()
    store.insert_memory_item({"id": "orphan1", "dataset_key": "ds_with_items", "raw_text": "x", "metadata": {}})
    store.insert_memory_item({"id": "orphan2", "dataset_key": "ds_with_items", "raw_text": "y", "metadata": {}})
    assert store.count_memory_items("ds_with_items") == 2
    client.delete("/datasets/ds_with_items")
    assert store.count_memory_items("ds_with_items", include_deleted=True) == 0


def test_delete_dataset_cascades_relationships(client):
    """Deleting a dataset removes relationships that reference it."""
    client.post("/datasets", json={**DATASET_PAYLOAD, "dataset_key": "cascade_src"})
    client.post("/datasets", json={**DATASET_PAYLOAD, "dataset_key": "cascade_tgt"})
    r_rel = client.post("/relationships", json={
        "id": "cascade_edge",
        "source_type": "dataset", "source_key": "cascade_src",
        "target_type": "dataset", "target_key": "cascade_tgt",
        "edge_type": "related",
    })
    assert r_rel.status_code == 200
    client.delete("/datasets/cascade_src")
    r_check = client.get("/relationships/cascade_edge")
    assert r_check.status_code == 404


def test_delete_tool_cascades_relationships(client):
    """Deleting a tool removes relationships that reference it."""
    client.post("/tools", json={**TOOL_PAYLOAD, "tool_key": "cascade_tool"})
    r_rel = client.post("/relationships", json={
        "id": "tool_cascade_edge",
        "source_type": "tool", "source_key": "cascade_tool",
        "target_type": "dataset", "target_key": "test_ds",
        "edge_type": "consumes",
    })
    assert r_rel.status_code == 200
    client.delete("/tools/cascade_tool")
    r_check = client.get("/relationships/tool_cascade_edge")
    assert r_check.status_code == 404


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
# Session-aware ingest middleware
# ---------------------------------------------------------------------------

def test_high_level_ingest_creates_main_session_and_raw_text(client):
    r = client.post("/ingest", json={"text": "remember this chat turn"})
    assert r.status_code == 200
    body = r.json()
    assert body["session"]["id"] == "main"
    assert body["message"]["session_id"] == "main"
    assert body["message"]["content"] == "remember this chat turn"
    assert body["message"]["raw_text_id"] == body["raw_text"]["id"]
    assert body["raw_text"]["text"] == "remember this chat turn"
    assert {"session", "message", "raw_text", "derived"} <= set(body)
    assert "trace" in body
    assert body["trace"]["chunks_written"] >= 1
    assert body["trace"]["graph_edges_written"] >= 1
    job_names = {job["name"] for job in body["derived"]}
    assert {"processor", "logic_analysis", "session_memory", "primitive_memory", "graph_edges", "candidate_observations"} <= job_names


def test_high_level_ingest_writes_logic_analysis_outputs(client):
    from app.store import get_store

    pack = {
        "key": "api_framework_logic",
        "display_name": "API Framework Logic",
        "primitive_rules": [
            {
                "kind": "framework",
                "pattern": r"\bMastra\b",
                "target_dataset_key": "api_framework_routes",
                "confidence": 0.82,
            }
        ],
    }
    target_dataset = {
        **DATASET_PAYLOAD,
        "dataset_key": "api_framework_routes",
        "display_name": "API Framework Routes",
        "semantic_description": "Framework route target used to prove /ingest does not write route candidates.",
    }
    assert client.post("/datasets", json=target_dataset).status_code == 200
    assert client.post("/ingest/rule-packs", json=pack).status_code == 200

    r = client.post(
        "/ingest",
        json={"session_id": "logic_api", "text": "TODO: compare Mastra for route handling."},
    )
    assert r.status_code == 200
    body = r.json()
    store = get_store()

    session_items = store.list_memory_items("session_memory")
    assert any(item["metadata"]["session_id"] == "logic_api" for item in session_items)
    assert any(item["metadata"]["raw_text_id"] == body["raw_text"]["id"] for item in session_items)
    assert any(item["metadata"]["session_message_id"] == body["message"]["id"] for item in session_items)
    assert any(item["metadata"]["logic_ingest"] is True for item in session_items)

    primitive_items = store.list_memory_items("ingest_primitives")
    framework_items = [
        item for item in primitive_items
        if item["metadata"].get("primitive_kind") == "framework" and item["raw_text"] == "Mastra"
    ]
    assert framework_items
    assert framework_items[0]["metadata"]["raw_text_id"] == body["raw_text"]["id"]
    assert framework_items[0]["metadata"]["session_message_id"] == body["message"]["id"]
    assert framework_items[0]["metadata"]["logic_ingest"] is True

    rels = store.list_relationships()
    assert any(rel["source_key"] == body["raw_text"]["id"] for rel in rels)
    assert any(rel["target_key"] == body["message"]["id"] for rel in rels)
    assert not any(rel["source_key"].startswith("raw-") and len(rel["source_key"]) == 20 for rel in rels)
    assert not any(rel["target_key"].startswith("msg-") and len(rel["target_key"]) == 20 for rel in rels)

    canonical_items = store.list_memory_items("api_framework_routes")
    assert any(item["metadata"].get("memory_role") == "canonical_entity" for item in canonical_items)
    derived_by_name = {job["name"]: job for job in body["derived"]}
    assert derived_by_name["primitive_memory"]["item_ids"]
    assert "api_framework_routes" in derived_by_name["logic_analysis"]["dataset_keys"]
    assert body["trace"]["route_targets"]
    assert any(route["dataset_key"] == "api_framework_routes" for route in body["trace"]["route_targets"])
    assert any(
        entity["dataset_key"] == "api_framework_routes" and entity["name"] == "Mastra"
        for entity in body["trace"]["canonical_entities"]
    )


def test_high_level_ingest_dedupes_repeated_entity_mentions_and_updates_canonical(client):
    from app.store import get_store

    pack = {
        "key": "canonical_framework_logic",
        "display_name": "Canonical Framework Logic",
        "namespace": "canonical_namespace",
        "primitive_rules": [
            {
                "kind": "framework",
                "pattern": r"\bMastra\b",
                "target_dataset_key": "canonical_frameworks",
                "confidence": 0.9,
            }
        ],
    }
    assert client.post("/ingest/rules", json=pack).status_code == 200

    first = client.post(
        "/ingest",
        json={
            "session_id": "canonical_one",
            "namespace": "canonical_namespace",
            "text": "Mastra is useful. Mastra handles workflows.",
        },
    )
    assert first.status_code == 200
    first_body = first.json()
    store = get_store()

    observations = [
        item for item in store.list_memory_items("ingest_primitives", limit=200)
        if item["metadata"].get("raw_text_id") == first_body["raw_text"]["id"]
        and item["metadata"].get("primitive_kind") == "framework"
    ]
    assert len(observations) == 1
    assert observations[0]["metadata"]["memory_role"] == "observation"
    assert observations[0]["metadata"]["mention_count"] == 2
    assert len(observations[0]["metadata"]["mentions"]) == 2
    assert first_body["trace"]["observations_written"] >= 1
    assert first_body["trace"]["observation_kinds"]["framework"] == 1
    assert first_body["trace"]["canonical_entities_upserted"] == 1
    trace_entity = first_body["trace"]["canonical_entities"][0]
    assert trace_entity["dataset_key"] == "canonical_frameworks"
    assert trace_entity["entity_kind"] == "framework"
    assert trace_entity["name"] == "Mastra"
    assert trace_entity["observation_id"] == observations[0]["id"]
    assert trace_entity["mention_count"] == 2

    canonical_items = store.list_memory_items("canonical_frameworks", limit=20)
    canonical_mastra = next(
        item for item in canonical_items
        if item["metadata"].get("canonical_name") == "Mastra"
    )
    assert canonical_mastra["metadata"]["memory_role"] == "canonical_entity"
    assert canonical_mastra["metadata"]["evidence_count"] == 1
    assert canonical_mastra["metadata"]["source_observation_ids"] == [observations[0]["id"]]

    second = client.post(
        "/ingest",
        json={
            "session_id": "canonical_two",
            "namespace": "canonical_namespace",
            "text": "Mastra should stay one canonical framework.",
        },
    )
    assert second.status_code == 200

    updated = store.get_memory_item(canonical_mastra["id"])
    assert updated is not None
    assert updated["metadata"]["evidence_count"] == 2
    assert len(updated["metadata"]["source_observation_ids"]) == 2
    assert set(updated["metadata"]["source_session_ids"]) == {"canonical_one", "canonical_two"}

    relationships = store.list_relationships()
    assert sum(
        1 for rel in relationships
        if rel["target_key"] == canonical_mastra["id"] and rel["edge_type"] == "shared_entity"
    ) == 2


def test_high_level_ingest_does_not_canonicalize_tasks(client):
    from app.store import get_store

    pack = {
        "key": "task_observation_logic",
        "display_name": "Task Observation Logic",
        "namespace": "task_namespace",
        "primitive_rules": [
            {
                "kind": "task",
                "pattern": r"\bTODO\b",
                "target_dataset_key": "task_targets_should_not_canonicalize",
                "confidence": 0.75,
            }
        ],
    }
    assert client.post("/ingest/rules", json=pack).status_code == 200
    response = client.post(
        "/ingest",
        json={
            "session_id": "task_noncanonical",
            "namespace": "task_namespace",
            "text": "TODO: write tests. TODO: review docs.",
        },
    )
    assert response.status_code == 200

    store = get_store()
    assert store.get_dataset("task_targets_should_not_canonicalize") is None
    task_observations = [
        item for item in store.list_memory_items("ingest_primitives", limit=200)
        if item["metadata"].get("session_id") == "task_noncanonical"
        and item["metadata"].get("primitive_kind") == "task"
    ]
    assert task_observations
    assert all(item["metadata"]["memory_role"] == "observation" for item in task_observations)
    body = response.json()
    assert body["trace"]["canonical_entities"] == []
    assert body["trace"]["canonical_entities_upserted"] == 0
    assert body["trace"]["observation_kinds"]["task"] == len(task_observations)


def test_high_level_ingest_derive_false_skips_logic_persistence(client):
    from app.store import get_store

    r = client.post(
        "/ingest",
        json={"session_id": "skip_logic", "text": "TODO: do not persist analyzer outputs.", "derive": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["derived"] == [
        {"name": "logic_analysis", "status": "skipped", "detail": "derive=false", "dataset_keys": [], "item_ids": []}
    ]
    store = get_store()
    assert not any(
        item["metadata"].get("session_id") == "skip_logic"
        for item in store.list_memory_items("session_memory")
    )


def test_high_level_ingest_uses_processor_sidecar_observations(client):
    from app.api.ingest import get_processor_service
    from app.schemas.processor import (
        ProcessorCandidate,
        ProcessorClassification,
        ProcessorEntity,
        ProcessorPhrase,
        ProcessorResponse,
        ProcessorSpan,
    )
    from app.store import get_store

    class FakeProcessor:
        provider = "sidecar"
        url = "http://fake-processor"
        strategy = "semantic"
        classify_enabled = True
        known_match_threshold = 0.7
        candidate_threshold = 0.4
        graceful_fallback = True

        def is_enabled(self):  # noqa: ANN201
            return True

        async def process_text(self, request):  # noqa: ANN001, ANN201
            text = request.text
            mastra_start = text.index("Mastra")
            return ProcessorResponse(
                processor="fake-spacy-minilm",
                processor_version="test/1",
                strategy="semantic",
                chunks=[
                    ProcessorSpan(
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        primitive="chunk",
                    )
                ],
                entities=[
                    ProcessorEntity(
                        text="Mastra",
                        label="PRODUCT",
                        char_start=mastra_start,
                        char_end=mastra_start + len("Mastra"),
                        confidence=0.91,
                    )
                ],
                phrases=[
                    ProcessorPhrase(
                        text="workflow framework",
                        label="noun_phrase",
                        char_start=text.index("workflow"),
                        char_end=text.index("framework") + len("framework"),
                        score=0.62,
                    )
                ],
                classifications=[
                    ProcessorClassification(
                        label="framework",
                        score=0.88,
                        matched_rule_key="framework_rules",
                        target_dataset_key="frameworks",
                    ),
                    ProcessorClassification(label="unknown_tooling_cluster", score=0.51),
                ],
                candidates=[
                    ProcessorCandidate(
                        label="workflow_tooling",
                        text="workflow framework",
                        char_start=text.index("workflow"),
                        char_end=text.index("framework") + len("framework"),
                        score=0.52,
                        suggested_dataset_key="workflow_tools",
                    )
                ],
            )

    client.app.dependency_overrides[get_processor_service] = lambda: FakeProcessor()
    try:
        r = client.post(
            "/ingest",
            json={"session_id": "sidecar_session", "text": "Mastra is a workflow framework."},
        )
        assert r.status_code == 200
    finally:
        client.app.dependency_overrides.pop(get_processor_service, None)

    store = get_store()
    primitives = store.list_memory_items("ingest_primitives")
    assert any(item["metadata"].get("source") == "processor_entity" for item in primitives)
    assert any(item["metadata"].get("source") == "processor_classification" for item in primitives)

    candidates = store.list_memory_items("ingest_candidates")
    assert any(item["metadata"].get("source") == "processor_phrase" for item in candidates)
    assert any(item["metadata"].get("source") == "processor_candidate" for item in candidates)
    assert any(item["metadata"].get("candidate_label") == "unknown_tooling_cluster" for item in candidates)


def test_ingest_rules_public_endpoint_stores_dataset_and_rule(client):
    rule = {
        "key": "public_framework_rules",
        "display_name": "Public Framework Rules",
        "semantic_rules": [
            {
                "kind": "framework",
                "target_dataset_key": "public_frameworks",
                "examples": ["Mastra orchestrates agent workflows."],
                "threshold": 0.78,
            }
        ],
        "entity_hints": [
            {
                "kind": "framework",
                "spacy_labels": ["PRODUCT", "ORG"],
                "noun_phrases": True,
                "target_dataset_key": "public_frameworks",
            }
        ],
        "primitive_rules": [
            {
                "kind": "framework_exact",
                "pattern": r"\bMastra\b",
                "target_dataset_key": "public_frameworks",
            }
        ],
        "datasets": [
            {
                "dataset_key": "public_frameworks",
                "display_name": "Public Frameworks",
                "schema_version": "v1",
                "semantic_description": "Framework knowledge created with public ingest rules.",
                "usage_guidance": "Use for framework observations.",
                "retrieval_capabilities": ["keyword"],
            }
        ],
    }
    r = client.post("/ingest/rules", json=rule)
    assert r.status_code == 200
    body = r.json()
    assert body["semantic_rules"][0]["kind"] == "framework"
    assert body["entity_hints"][0]["spacy_labels"] == ["PRODUCT", "ORG"]

    dataset = client.get("/datasets/public_frameworks")
    assert dataset.status_code == 200
    assert dataset.json()["display_name"] == "Public Frameworks"


def test_high_level_ingest_appends_to_existing_session_history(client):
    first = client.post("/ingest", json={"session_id": "planning", "text": "first turn"})
    assert first.status_code == 200
    second = client.post("/ingest", json={"session_id": "planning", "role": "assistant", "text": "second turn"})
    assert second.status_code == 200

    history = client.get("/sessions/planning/history")
    assert history.status_code == 200
    assert [m["content"] for m in history.json()] == ["first turn", "second turn"]
    assert [m["role"] for m in history.json()] == ["user", "assistant"]


def test_high_level_ingest_compacts_autocontext_when_threshold_is_exceeded(client):
    for i in range(4):
        r = client.post(
            "/ingest",
            json={
                "session_id": "compact_me",
                "text": f"message {i} " + ("x" * 80),
                "max_context_tokens": 30,
                "summary_target_tokens": 20,
            },
        )
        assert r.status_code == 200

    full = client.get("/sessions/compact_me/history")
    assert full.status_code == 200
    assert len(full.json()) == 4
    assert any(not m["autocontext_enabled"] for m in full.json())

    compact = client.get("/sessions/compact_me/history?autocontext_only=true")
    assert compact.status_code == 200
    assert len(compact.json()) < 4

    context = client.post("/context", json={"session_id": "compact_me", "prompt": "message"})
    assert context.status_code == 200
    assert len(context.json()["summaries"]) >= 1


def test_context_package_uses_dataset_memory_items(client):
    from app.store import get_store
    store = get_store()
    store.insert_memory_item({
        "id": "ctx_item_1",
        "dataset_key": "test_ds",
        "raw_text": "context package keyword",
        "metadata": {"kind": "note"},
    })
    r = client.post("/context", json={"prompt": "keyword", "dataset_keys": ["test_ds"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_keys"] == ["test_ds"]
    assert any(item["source_id"] == "ctx_item_1" for item in body["items"])


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


def test_ingest_text_requires_embedding(client):
    r = client.post(
        "/datasets/test_ds/ingest/text",
        json={"text": "hello\n\nworld", "max_chars": 10, "overlap_chars": 2},
    )
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


def test_soft_delete_idempotent_returns_404(client):
    """Second soft-delete of the same item should return 404."""
    from app.store import get_store
    get_store().insert_memory_item({
        "id": "double_del",
        "dataset_key": "test_ds",
        "raw_text": "delete me twice",
        "metadata": {},
    })
    r1 = client.delete("/datasets/test_ds/items/double_del")
    assert r1.status_code == 200
    r2 = client.delete("/datasets/test_ds/items/double_del")
    assert r2.status_code == 404


def test_get_item_soft_deleted_returns_404(client):
    """GET /items/{id} returns 404 for soft-deleted items by default."""
    from app.store import get_store
    get_store().insert_memory_item({
        "id": "get_del_test",
        "dataset_key": "test_ds",
        "raw_text": "soft deleted item",
        "metadata": {},
    })
    client.delete("/datasets/test_ds/items/get_del_test")
    r = client.get("/datasets/test_ds/items/get_del_test")
    assert r.status_code == 404
    r2 = client.get("/datasets/test_ds/items/get_del_test?include_deleted=true")
    assert r2.status_code == 200
    assert r2.json()["is_deleted"] is True


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
# Namespaces
# ---------------------------------------------------------------------------

def test_namespace_tools_are_isolated(client):
    r_create = client.post("/create_namespace", json={"namespace": "chat_project"})
    assert r_create.status_code == 200
    assert r_create.json()["api_root"] == "/chat_project"

    r_tool = client.post(
        "/chat_project/tools",
        json={
            "tool_key": "chat_tool",
            "name": "Chat Tool",
            "description": "A namespace-specific tool",
            "status": "active",
        },
    )
    assert r_tool.status_code == 200

    r_namespace_tools = client.get("/chat_project/tools")
    assert r_namespace_tools.status_code == 200
    assert "chat_tool" in [t["tool_key"] for t in r_namespace_tools.json()]

    r_default_tools = client.get("/tools")
    assert r_default_tools.status_code == 200
    assert "chat_tool" not in [t["tool_key"] for t in r_default_tools.json()]

    r_namespaces = client.get("/namespaces")
    assert r_namespaces.status_code == 200
    assert "chat_project" in r_namespaces.json()


def test_namespace_subspace_tools_are_isolated(client):
    r_namespace = client.post("/create_namespace", json={"namespace": "book_app"})
    assert r_namespace.status_code == 200

    r_dev = client.post("/book_app/new_subspace", json={"subspace": "dev"})
    assert r_dev.status_code == 200
    assert r_dev.json()["api_root"] == "/book_app/dev"

    r_prod = client.post("/book_app/new_subspace", json={"subspace": "prod"})
    assert r_prod.status_code == 200
    assert r_prod.json()["api_root"] == "/book_app/prod"

    parent_tool = {
        "tool_key": "parent_tool",
        "name": "Parent Tool",
        "description": "A namespace-level tool",
        "status": "active",
    }
    dev_tool = {
        "tool_key": "dev_tool",
        "name": "Dev Tool",
        "description": "A dev subspace tool",
        "status": "active",
    }
    prod_tool = {
        "tool_key": "prod_tool",
        "name": "Prod Tool",
        "description": "A prod subspace tool",
        "status": "active",
    }

    assert client.post("/book_app/tools", json=parent_tool).status_code == 200
    assert client.post("/book_app/dev/tools", json=dev_tool).status_code == 200
    assert client.post("/book_app/prod/tools", json=prod_tool).status_code == 200

    parent_keys = [t["tool_key"] for t in client.get("/book_app/tools").json()]
    dev_keys = [t["tool_key"] for t in client.get("/book_app/dev/tools").json()]
    prod_keys = [t["tool_key"] for t in client.get("/book_app/prod/tools").json()]

    assert "parent_tool" in parent_keys
    assert "dev_tool" not in parent_keys
    assert "prod_tool" not in parent_keys

    assert dev_keys == ["dev_tool"]
    assert prod_keys == ["prod_tool"]

    r_subspaces = client.get("/book_app/subspaces")
    assert r_subspaces.status_code == 200
    assert r_subspaces.json() == ["dev", "prod"]


def test_namespace_and_subspace_docs_are_scoped(client):
    client.post("/create_namespace", json={"namespace": "docs_app"})
    client.post("/docs_app/new_subspace", json={"subspace": "dev"})

    r_namespace_docs = client.get("/docs_app/docs")
    assert r_namespace_docs.status_code == 200
    assert "swagger-ui" in r_namespace_docs.text

    r_namespace_openapi = client.get("/docs_app/openapi.json")
    assert r_namespace_openapi.status_code == 200
    assert r_namespace_openapi.json()["servers"] == [{"url": "/docs_app"}]

    r_subspace_docs = client.get("/docs_app/dev/docs")
    assert r_subspace_docs.status_code == 200
    assert "swagger-ui" in r_subspace_docs.text

    r_subspace_openapi = client.get("/docs_app/dev/openapi.json")
    assert r_subspace_openapi.status_code == 200
    assert r_subspace_openapi.json()["servers"] == [{"url": "/docs_app/dev"}]


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
    from app import __version__
    assert body["result"]["serverInfo"]["version"] == __version__


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
