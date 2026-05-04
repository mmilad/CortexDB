"""Simulated end-to-end CortexDB usage via HTTP API with embedding disabled."""
from __future__ import annotations

import os
import tempfile

# Keep this test module fully local/deterministic (no external embedding calls).
os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"

import pytest
from fastapi.testclient import TestClient

SIM_PREFIX = "sim_usage"
DATASET_A_KEY = f"{SIM_PREFIX}_dataset_alpha"
DATASET_B_KEY = f"{SIM_PREFIX}_dataset_beta"
RELATIONSHIP_ID = f"{SIM_PREFIX}_relationship_alpha_beta"
ITEM_ID = f"{SIM_PREFIX}_item_alpha_001"
KEYWORD_TOKEN = f"{SIM_PREFIX}_keyword_token_001"

DATASET_A_PAYLOAD = {
    "dataset_key": DATASET_A_KEY,
    "display_name": "Simulated Usage Alpha Dataset",
    "schema_version": "v1",
    "semantic_description": "Primary dataset for simulated end-to-end usage testing.",
    "usage_guidance": "Use for deterministic API workflow validation.",
    "retrieval_capabilities": ["keyword", "filter_only"],
    "llm_summary": "Contains simulated alpha memory records.",
}

DATASET_B_PAYLOAD = {
    "dataset_key": DATASET_B_KEY,
    "display_name": "Simulated Usage Beta Dataset",
    "schema_version": "v1",
    "semantic_description": "Secondary dataset used for graph relationship traversal tests.",
    "usage_guidance": "Use as the target node in simulated graph exploration.",
    "retrieval_capabilities": ["filter_only"],
    "llm_summary": "Represents a related dataset in the simulated graph.",
}


@pytest.fixture(scope="module")
def client():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    os.environ["CORTEXDB_DB_PATH"] = db_path

    # Import after env vars are set so singletons pick them up.
    import app.embed.service as embed_mod
    import app.store.main as store_mod

    # Reset singletons so this module has a fresh isolated state.
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


def test_simulated_usage_end_to_end_http_api(client: TestClient) -> None:
    # 1) Register two datasets through the public HTTP API.
    r_create_a = client.post("/datasets", json=DATASET_A_PAYLOAD)
    assert r_create_a.status_code == 200
    assert r_create_a.json()["dataset_key"] == DATASET_A_KEY

    r_create_b = client.post("/datasets", json=DATASET_B_PAYLOAD)
    assert r_create_b.status_code == 200
    assert r_create_b.json()["dataset_key"] == DATASET_B_KEY

    # 2) Create a dataset->dataset relationship.
    r_rel = client.post(
        "/relationships",
        json={
            "id": RELATIONSHIP_ID,
            "source_type": "dataset",
            "source_key": DATASET_A_KEY,
            "target_type": "dataset",
            "target_key": DATASET_B_KEY,
            "edge_type": "related",
            "description": "Simulated usage path from alpha to beta dataset.",
        },
    )
    assert r_rel.status_code == 200
    assert r_rel.json()["id"] == RELATIONSHIP_ID

    # 3) Add memory directly through the store.
    # Ingest is intentionally unavailable when CORTEXDB_EMBED_PROVIDER=none.
    from app.store import get_store

    get_store().insert_memory_item(
        {
            "id": ITEM_ID,
            "dataset_key": DATASET_A_KEY,
            "raw_text": f"Simulated record containing {KEYWORD_TOKEN} for keyword search.",
            "metadata": {"source": "simulated_usage_test"},
        }
    )

    # 4) Run keyword-only search and assert mode + semantic hit.
    r_search = client.post(
        f"/datasets/{DATASET_A_KEY}/search",
        json={
            "query": "unused in keyword mode",
            "keyword_query": KEYWORD_TOKEN,
            "vector_weight": 0.0,
            "top_k": 5,
        },
    )
    assert r_search.status_code == 200
    search_body = r_search.json()
    assert search_body["search_mode"] == "keyword"
    assert len(search_body["hits"]) == 1
    assert search_body["hits"][0]["item"]["id"] == ITEM_ID

    # 5) Explore graph from dataset A and assert expected topology shape.
    r_graph = client.get(f"/graph/explore?start=dataset:{DATASET_A_KEY}&depth=1")
    assert r_graph.status_code == 200
    graph_body = r_graph.json()
    assert graph_body["root"] == DATASET_A_KEY
    assert isinstance(graph_body.get("nodes"), list)
    assert isinstance(graph_body.get("edges"), list)
    assert any(n["key"] == DATASET_B_KEY for n in graph_body["nodes"])
    assert any(
        {edge["source"], edge["target"]} == {DATASET_A_KEY, DATASET_B_KEY}
        for edge in graph_body["edges"]
    )

    # 6) Validate dataset metadata and assert validation timestamp is set.
    r_validate = client.post(f"/datasets/{DATASET_A_KEY}/validate")
    assert r_validate.status_code == 200
    validate_body = r_validate.json()
    assert validate_body["dataset_key"] == DATASET_A_KEY
    assert validate_body["last_validated_at"] is not None

    # 7) Soft-delete item and verify list behavior with/without include_deleted.
    r_delete = client.delete(f"/datasets/{DATASET_A_KEY}/items/{ITEM_ID}")
    assert r_delete.status_code == 200
    assert r_delete.json()["soft_deleted"] == ITEM_ID

    r_list_default = client.get(f"/datasets/{DATASET_A_KEY}/items")
    assert r_list_default.status_code == 200
    assert ITEM_ID not in [item["id"] for item in r_list_default.json()]

    r_list_with_deleted = client.get(f"/datasets/{DATASET_A_KEY}/items?include_deleted=true")
    assert r_list_with_deleted.status_code == 200
    assert ITEM_ID in [item["id"] for item in r_list_with_deleted.json()]
