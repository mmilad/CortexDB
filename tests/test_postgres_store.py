"""Opt-in PostgreSQL parity checks.

These tests stay skipped in the normal SQLite-only test run. Set
CORTEXDB_DATABASE_URL and install the ``postgres`` extra to exercise the real
pgvector backend against a running compose service.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest


@pytest.mark.skipif(not os.environ.get("CORTEXDB_DATABASE_URL"), reason="PostgreSQL integration is opt-in")
def test_postgres_store_scopes_hybrid_records_and_cascades() -> None:
    try:
        import psycopg  # noqa: F401
        import pgvector  # noqa: F401
    except ImportError:
        pytest.skip("install the postgres extra to run PostgreSQL integration tests")

    from app.store import PostgresStore

    schema = f"cortexdb_test_{uuid.uuid4().hex[:12]}"
    assert re.fullmatch(r"[a-z0-9_]+", schema)
    store = PostgresStore(os.environ["CORTEXDB_DATABASE_URL"], schema=schema)
    dataset_key = f"test_{uuid.uuid4().hex}"
    try:
        store.upsert_dataset(dataset_key, {"display_name": "Postgres test"})
        store.upsert_relationship({
            "id": f"{dataset_key}_relation",
            "source_type": "dataset",
            "source_key": dataset_key,
            "target_type": "dataset",
            "target_key": dataset_key,
            "edge_type": "related",
        })
        relationship = store.get_relationship(f"{dataset_key}_relation")
        assert relationship is not None
        assert isinstance(relationship["created_at"], str)
        store.insert_memory_item({
            "id": f"{dataset_key}_alice",
            "dataset_key": dataset_key,
            "raw_text": "alice private release note",
            "metadata": {"kind": "note"},
            "scope": {"kind": "personal", "owner_id": "alice"},
        })
        store.insert_memory_item({
            "id": f"{dataset_key}_bob",
            "dataset_key": dataset_key,
            "raw_text": "bob private release note",
            "metadata": {"kind": "note"},
            "scope": {"kind": "personal", "owner_id": "bob"},
        })

        rows = store.search_memory_items(
            dataset_key,
            query_vector=None,
            keyword_query="private release",
            vector_weight=0.0,
            access={"principal_id": "alice", "include_global": True},
        )
        assert [row["id"] for row in rows] == [f"{dataset_key}_alice"]

        assert store.delete_dataset(dataset_key) is True
        assert store.get_memory_item(f"{dataset_key}_alice") is None
    finally:
        with store._conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
        store._conn.commit()  # type: ignore[attr-defined]
        store.close()
