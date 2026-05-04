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


def test_delete_dataset_cascades_relationships(store):
    store.upsert_dataset("src_ds", {})
    store.upsert_dataset("tgt_ds", {})
    store.upsert_relationship({
        "id": "edge1", "source_type": "dataset", "source_key": "src_ds",
        "target_type": "dataset", "target_key": "tgt_ds",
        "edge_type": "related", "join_fields": [], "description": "",
    })
    store.upsert_relationship({
        "id": "edge2", "source_type": "dataset", "source_key": "other_ds",
        "target_type": "dataset", "target_key": "src_ds",
        "edge_type": "feeds_into", "join_fields": [], "description": "",
    })
    store.delete_dataset("src_ds")
    assert store.get_relationship("edge1") is None
    assert store.get_relationship("edge2") is None


def test_delete_tool_cascades_relationships(store):
    store.upsert_tool("my_tool", {})
    store.upsert_relationship({
        "id": "tedge1", "source_type": "tool", "source_key": "my_tool",
        "target_type": "dataset", "target_key": "some_ds",
        "edge_type": "consumes", "join_fields": [], "description": "",
    })
    store.delete_tool("my_tool")
    assert store.get_relationship("tedge1") is None


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


def test_hard_delete_missing_returns_false(store):
    """delete_memory_item returns False when the item does not exist."""
    assert store.delete_memory_item("does-not-exist") is False


def test_hard_delete_cascades_relationships(store):
    """Hard-deleting an item must cascade-delete edges that reference it."""
    store.upsert_dataset("cascade_ds", {})
    store.upsert_dataset("other_ds", {})
    store.insert_memory_item({"id": "ci_item", "dataset_key": "cascade_ds", "raw_text": "x", "metadata": {}})

    store.upsert_relationship({
        "id": "ci_edge1", "source_type": "memory_item", "source_key": "ci_item",
        "target_type": "dataset", "target_key": "other_ds",
        "edge_type": "related", "join_fields": [], "description": "",
    })
    store.upsert_relationship({
        "id": "ci_edge2", "source_type": "dataset", "source_key": "cascade_ds",
        "target_type": "memory_item", "target_key": "ci_item",
        "edge_type": "contains", "join_fields": [], "description": "",
    })
    # Edge unrelated to ci_item — must survive.
    store.upsert_relationship({
        "id": "ci_edge3", "source_type": "dataset", "source_key": "cascade_ds",
        "target_type": "dataset", "target_key": "other_ds",
        "edge_type": "feeds_into", "join_fields": [], "description": "",
    })

    assert store.delete_memory_item("ci_item") is True
    assert store.get_relationship("ci_edge1") is None
    assert store.get_relationship("ci_edge2") is None
    assert store.get_relationship("ci_edge3") is not None


def test_soft_delete_preserves_relationships(store):
    """Soft-deleting an item must NOT delete its relationships (item is recoverable)."""
    store.upsert_dataset("soft_ds", {})
    store.upsert_dataset("soft_other", {})
    store.insert_memory_item({"id": "soft_item", "dataset_key": "soft_ds", "raw_text": "y", "metadata": {}})
    store.upsert_relationship({
        "id": "soft_edge", "source_type": "memory_item", "source_key": "soft_item",
        "target_type": "dataset", "target_key": "soft_other",
        "edge_type": "related", "join_fields": [], "description": "",
    })

    assert store.soft_delete_memory_item("soft_item") is True
    assert store.get_relationship("soft_edge") is not None


def test_delete_dataset_cascades_item_relationships(store):
    """Deleting a dataset must also remove relationships owned by its memory items."""
    store.upsert_dataset("item_rel_ds", {})
    store.upsert_dataset("peer_ds", {})
    store.insert_memory_item({"id": "ir_item", "dataset_key": "item_rel_ds", "raw_text": "z", "metadata": {}})
    store.upsert_relationship({
        "id": "ir_edge", "source_type": "memory_item", "source_key": "ir_item",
        "target_type": "dataset", "target_key": "peer_ds",
        "edge_type": "related", "join_fields": [], "description": "",
    })

    store.delete_dataset("item_rel_ds")
    assert store.get_relationship("ir_edge") is None


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
    # BM25: h1 contains "relevant" → positive score; h2 does not → zero score
    assert results[0]["keyword_score"] > 0.0
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


# ---------------------------------------------------------------------------
# BM25 keyword scoring
# ---------------------------------------------------------------------------

def test_bm25_keyword_only_positive_score(store):
    """Items containing the query term receive a BM25 score > 0."""
    from app.store.search import bm25_score
    items = [
        {"id": "a", "raw_text": "the quick brown fox"},
        {"id": "b", "raw_text": "lazy dog"},
    ]
    scored = bm25_score(items, "fox")
    a = next(i for i in scored if i["id"] == "a")
    b = next(i for i in scored if i["id"] == "b")
    assert a["keyword_score"] > 0.0
    assert b["keyword_score"] == 0.0


def test_bm25_multi_term(store):
    """Multi-term queries accumulate IDF contributions."""
    from app.store.search import bm25_score
    items = [
        {"id": "a", "raw_text": "machine learning model training"},
        {"id": "b", "raw_text": "machine learning inference"},
        {"id": "c", "raw_text": "totally unrelated text here"},
    ]
    scored = bm25_score(items, "machine learning training")
    by_id = {i["id"]: i["keyword_score"] for i in scored}
    # "a" has all three terms; "b" has two; "c" has none
    assert by_id["a"] > by_id["b"] > by_id["c"]
    assert by_id["c"] == 0.0


def test_bm25_normalised_to_unit(store):
    """BM25 normalized score must be in [0, 1]."""
    from app.store.search import bm25_score
    items = [{"id": str(i), "raw_text": f"word{i} word{i} word{i}"} for i in range(10)]
    scored = bm25_score(items, "word0 word1 word2")
    for it in scored:
        assert 0.0 <= it["keyword_score"] <= 1.0


# ---------------------------------------------------------------------------
# sqlite-vec ANN index
# ---------------------------------------------------------------------------

def test_vec_enabled_flag(store):
    """Vec is enabled when sqlite-vec is installed."""
    # sqlite-vec is installed in the dev environment
    assert store.vec_enabled is True


def test_vec_table_created_on_first_ingest(store):
    """Inserting an item with an embedding creates the vec0 table."""
    store.upsert_dataset("vec_ds", {})
    store.insert_memory_item({
        "id": "vv1", "dataset_key": "vec_ds", "raw_text": "hello",
        "metadata": {}, "embedding": [1.0, 0.0, 0.0, 0.0], "embedding_model": "m",
    })
    from app.store.vec import table_exists, get_dim
    assert table_exists(store._conn, "vec_ds")
    assert get_dim(store._conn, "vec_ds") == 4


def test_vec_search_returns_correct_order(store):
    """ANN search via vec0 returns the nearest item first."""
    store.upsert_dataset("vec_order", {})
    store.insert_memory_item({
        "id": "vo1", "dataset_key": "vec_order", "raw_text": "near",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    store.insert_memory_item({
        "id": "vo2", "dataset_key": "vec_order", "raw_text": "far",
        "metadata": {}, "embedding": [0.0, 1.0], "embedding_model": "m",
    })
    results = store.search_memory_items("vec_order", query_vector=[1.0, 0.0], top_k=2)
    assert results[0]["id"] == "vo1"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_vec_hard_delete_removes_vec_entry(store):
    """Hard-deleting an item removes it from the vec0 table."""
    from app.store.vec import vec_table_name
    store.upsert_dataset("vec_del", {})
    store.insert_memory_item({
        "id": "vd1", "dataset_key": "vec_del", "raw_text": "to delete",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    tbl = vec_table_name("vec_del")
    count_before = store._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    store.delete_memory_item("vd1")
    count_after = store._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    assert count_before == 1
    assert count_after == 0


def test_vec_dataset_delete_drops_table(store):
    """Deleting a dataset drops its vec0 table."""
    from app.store.vec import table_exists
    store.upsert_dataset("vec_drop", {})
    store.insert_memory_item({
        "id": "vdr1", "dataset_key": "vec_drop", "raw_text": "x",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    assert table_exists(store._conn, "vec_drop")
    store.delete_dataset("vec_drop")
    assert not table_exists(store._conn, "vec_drop")


def test_rebuild_vec_index(store):
    """rebuild_vec_index repopulates the vec0 table from stored embeddings."""
    from app.store.vec import table_exists, vec_table_name
    store.upsert_dataset("vec_rebuild", {})
    store.insert_memory_item({
        "id": "vrb1", "dataset_key": "vec_rebuild", "raw_text": "a",
        "metadata": {}, "embedding": [1.0, 0.0], "embedding_model": "m",
    })
    store.insert_memory_item({
        "id": "vrb2", "dataset_key": "vec_rebuild", "raw_text": "b",
        "metadata": {}, "embedding": [0.0, 1.0], "embedding_model": "m",
    })
    tbl = vec_table_name("vec_rebuild")
    store._conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    store._conn.execute("UPDATE datasets SET vec_dim = NULL WHERE dataset_key = 'vec_rebuild'")
    store._conn.commit()
    assert not table_exists(store._conn, "vec_rebuild")

    count = store.rebuild_vec_index("vec_rebuild")
    assert count == 2
    assert table_exists(store._conn, "vec_rebuild")
    # Confirm search works after rebuild
    results = store.search_memory_items("vec_rebuild", query_vector=[1.0, 0.0], top_k=2)
    assert results[0]["id"] == "vrb1"
