"""Unit tests for SqliteStore."""
from __future__ import annotations

import tempfile
import os
import pytest

from app.store import SqliteStore, cosine_similarity


@pytest.fixture()
def store():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    s = SqliteStore(path)
    yield s
    s.close()
    os.unlink(path)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

def test_upsert_and_get_dataset(store):
    store.upsert_dataset("ds1", {"dataset_key": "ds1", "display_name": "Test"})
    d = store.get_dataset("ds1")
    assert d is not None
    assert d["dataset_key"] == "ds1"


def test_list_datasets_empty(store):
    assert store.list_datasets() == {}


def test_list_datasets(store):
    store.upsert_dataset("a", {"k": "a"})
    store.upsert_dataset("b", {"k": "b"})
    lst = store.list_datasets()
    assert set(lst.keys()) == {"a", "b"}


def test_delete_dataset_existing(store):
    store.upsert_dataset("x", {})
    assert store.delete_dataset("x") is True
    assert store.get_dataset("x") is None


def test_delete_dataset_missing(store):
    assert store.delete_dataset("nope") is False


def test_delete_dataset_cascades_memory_items(store):
    store.upsert_dataset("del_ds", {})
    store.insert_memory_item({"id": "di1", "dataset_key": "del_ds", "raw_text": "a", "metadata": {}})
    store.insert_memory_item({"id": "di2", "dataset_key": "del_ds", "raw_text": "b", "metadata": {}})
    assert store.count_memory_items("del_ds") == 2
    store.delete_dataset("del_ds")
    assert store.count_memory_items("del_ds", include_deleted=True) == 0


def test_dataset_embedding_roundtrip(store):
    store.upsert_dataset("e1", {})
    store.set_dataset_embedding("e1", "hello", [0.1, 0.2], "test/model")
    vec, model = store.get_dataset_embedding("e1")
    assert vec == pytest.approx([0.1, 0.2])
    assert model == "test/model"


def test_dataset_embedding_missing(store):
    store.upsert_dataset("e2", {})
    vec, model = store.get_dataset_embedding("e2")
    assert vec is None
    assert model is None


# ---------------------------------------------------------------------------
# Tool CRUD
# ---------------------------------------------------------------------------

def test_upsert_and_get_tool(store):
    store.upsert_tool("t1", {"tool_key": "t1", "name": "My Tool"})
    t = store.get_tool("t1")
    assert t is not None
    assert t["tool_key"] == "t1"


def test_delete_tool(store):
    store.upsert_tool("t2", {})
    assert store.delete_tool("t2") is True
    assert store.get_tool("t2") is None


# ---------------------------------------------------------------------------
# Relationship CRUD
# ---------------------------------------------------------------------------

def test_upsert_and_get_relationship(store):
    rel = {
        "id": "r1",
        "source_type": "dataset",
        "source_key": "ds1",
        "target_type": "dataset",
        "target_key": "ds2",
        "edge_type": "related",
        "join_fields": [],
        "description": "related",
    }
    store.upsert_relationship(rel)
    got = store.get_relationship("r1")
    assert got is not None
    assert got["source_key"] == "ds1"
    assert got["join_fields"] == []


def test_list_relationships_filter(store):
    for i, (src, tgt) in enumerate([("a", "b"), ("c", "d"), ("a", "c")]):
        store.upsert_relationship({
            "id": f"r{i}", "source_type": "dataset", "source_key": src,
            "target_type": "dataset", "target_key": tgt,
            "edge_type": "related", "join_fields": [], "description": "",
        })
    results = store.list_relationships(source_key="a")
    assert len(results) == 2


def test_delete_relationship(store):
    store.upsert_relationship({
        "id": "rx", "source_type": "dataset", "source_key": "x",
        "target_type": "dataset", "target_key": "y",
        "edge_type": "related", "join_fields": [], "description": "",
    })
    assert store.delete_relationship("rx") is True
    assert store.get_relationship("rx") is None


# ---------------------------------------------------------------------------
# Memory items
# ---------------------------------------------------------------------------

def test_insert_and_get_memory_item(store):
    store.upsert_dataset("ds", {})
    store.insert_memory_item({
        "id": "m1", "dataset_key": "ds", "raw_text": "hello world",
        "metadata": {"tag": "test"}, "embedding": [1.0, 0.0], "embedding_model": "x",
    })
    item = store.get_memory_item("m1")
    assert item is not None
    assert item["raw_text"] == "hello world"
    assert item["metadata"]["tag"] == "test"
    assert item["is_deleted"] is False


def test_soft_delete(store):
    store.upsert_dataset("ds", {})
    store.insert_memory_item({"id": "m2", "dataset_key": "ds", "raw_text": "bye", "metadata": {}})
    assert store.soft_delete_memory_item("m2") is True
    # Should still be retrievable by direct get (no filter on is_deleted there)
    item = store.get_memory_item("m2")
    assert item is not None
    assert item["is_deleted"] is True


def test_soft_deleted_excluded_from_list(store):
    store.upsert_dataset("ds", {})
    store.insert_memory_item({"id": "ma", "dataset_key": "ds", "raw_text": "visible", "metadata": {}})
    store.insert_memory_item({"id": "mb", "dataset_key": "ds", "raw_text": "hidden", "metadata": {}})
    store.soft_delete_memory_item("mb")

    visible = store.list_memory_items("ds")
    assert len(visible) == 1
    assert visible[0]["id"] == "ma"

    all_items = store.list_memory_items("ds", include_deleted=True)
    assert len(all_items) == 2


def test_hard_delete(store):
    store.upsert_dataset("ds", {})
    store.insert_memory_item({"id": "mc", "dataset_key": "ds", "raw_text": "gone", "metadata": {}})
    assert store.delete_memory_item("mc") is True
    assert store.get_memory_item("mc") is None


def test_count_items(store):
    store.upsert_dataset("ds", {})
    for i in range(3):
        store.insert_memory_item({
            "id": f"ci{i}", "dataset_key": "ds", "raw_text": f"text {i}", "metadata": {},
        })
    store.soft_delete_memory_item("ci0")
    assert store.count_memory_items("ds") == 2
    assert store.count_memory_items("ds", include_deleted=True) == 3


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def test_vector_search(store):
    store.upsert_dataset("vs", {})
    store.insert_memory_item({
        "id": "v1", "dataset_key": "vs", "raw_text": "match",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    store.insert_memory_item({
        "id": "v2", "dataset_key": "vs", "raw_text": "no match",
        "metadata": {}, "embedding": [0.0, 1.0], "embedding_model": "m",
    })
    results = store.search_memory_items("vs", query_vector=[1.0, 0.0], top_k=2)
    assert results[0]["id"] == "v1"
    assert results[0]["score"] == pytest.approx(1.0)


def test_keyword_only_search(store):
    store.upsert_dataset("ks", {})
    store.insert_memory_item({"id": "k1", "dataset_key": "ks", "raw_text": "Hello World", "metadata": {}})
    store.insert_memory_item({"id": "k2", "dataset_key": "ks", "raw_text": "Goodbye Moon", "metadata": {}})
    results = store.search_memory_items("ks", query_vector=None, keyword_query="hello", vector_weight=0.0)
    assert len(results) == 1
    assert results[0]["id"] == "k1"


def test_hybrid_search(store):
    store.upsert_dataset("hs", {})
    store.insert_memory_item({
        "id": "h1", "dataset_key": "hs", "raw_text": "relevant keyword",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    store.insert_memory_item({
        "id": "h2", "dataset_key": "hs", "raw_text": "no keyword",
        "metadata": {}, "embedding": [0.9, 0.1], "embedding_model": "m",
    })
    # hybrid: vector_weight=0.5, keyword "relevant"
    results = store.search_memory_items(
        "hs", query_vector=[1.0, 0.0], keyword_query="relevant", vector_weight=0.5, top_k=2
    )
    assert results[0]["id"] == "h1"
    assert results[0]["keyword_score"] == 1.0
    assert results[1]["keyword_score"] == 0.0


def test_metadata_filter(store):
    store.upsert_dataset("mf", {})
    store.insert_memory_item({
        "id": "mf1", "dataset_key": "mf", "raw_text": "item a",
        "metadata": {"status": "open"}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    store.insert_memory_item({
        "id": "mf2", "dataset_key": "mf", "raw_text": "item b",
        "metadata": {"status": "closed"}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    results = store.search_memory_items(
        "mf", query_vector=[1.0, 0.0], metadata_filters={"status": "open"}
    )
    assert len(results) == 1
    assert results[0]["id"] == "mf1"
