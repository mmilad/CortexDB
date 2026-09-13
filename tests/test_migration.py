"""Deterministic checks for the SQLite-to-PostgreSQL migration contract."""

from __future__ import annotations

from app.migrate import migrate_sqlite_to_postgres
from app.store import SqliteStore


def test_migration_is_idempotent_and_preserves_stable_records(tmp_path) -> None:
    source = SqliteStore(str(tmp_path / "source.sqlite"))
    target = SqliteStore(str(tmp_path / "target.sqlite"))
    try:
        source.upsert_dataset("project-plan", {"display_name": "Project plan"})
        source.set_dataset_embedding("project-plan", "project plan", [1.0, 0.0], "fixture/embed")
        source.upsert_tool("read-file", {"name": "Read file"})
        source.upsert_ingest_rule_pack("default", {"status": "active", "rules": []})
        source.upsert_relationship({
            "id": "dataset-edge",
            "source_type": "dataset",
            "source_key": "project-plan",
            "target_type": "tool",
            "target_key": "read-file",
            "edge_type": "uses",
        })
        source.upsert_session({"id": "assistant", "metadata": {"owner": "alice"}})
        source.insert_raw_text({"id": "raw-1", "text": "A source paragraph", "metadata": {"page": 1}})
        source.insert_session_summary({
            "id": "summary-1",
            "session_id": "assistant",
            "summary": "A summary",
            "message_ids": ["message-1"],
        })
        source.insert_session_message({
            "id": "message-1",
            "session_id": "assistant",
            "role": "user",
            "content": "Remember this.",
            "raw_text_id": "raw-1",
            "summary_id": "summary-1",
        })
        source.insert_memory_item({
            "id": "memory-1",
            "dataset_key": "project-plan",
            "raw_text": "The release is Friday.",
            "metadata": {"source": "user"},
            "scope": {"kind": "personal", "owner_id": "alice"},
            "embedding": [1.0, 0.0],
            "embedding_model": "fixture/embed",
        })

        first_source, first_target = migrate_sqlite_to_postgres(source, target)
        second_source, second_target = migrate_sqlite_to_postgres(source, target)

        assert first_source == first_target == second_source == second_target
        assert target.get_dataset("project-plan")["display_name"] == "Project plan"
        assert target.get_dataset_embedding("project-plan") == ([1.0, 0.0], "fixture/embed")
        assert target.get_raw_text("raw-1")["text"] == "A source paragraph"
        assert target.get_session("assistant")["metadata"] == {"owner": "alice"}
        assert target.list_session_messages("assistant")[0]["id"] == "message-1"
        assert target.list_session_summaries("assistant")[0]["id"] == "summary-1"
        assert target.get_memory_item("memory-1")["scope"]["owner_id"] == "alice"
    finally:
        source.close()
        target.close()


def test_migration_merge_allows_existing_target_records(tmp_path) -> None:
    source = SqliteStore(str(tmp_path / "source-merge.sqlite"))
    target = SqliteStore(str(tmp_path / "target-merge.sqlite"))
    try:
        source.upsert_dataset("project-plan", {"display_name": "Project plan"})
        target.upsert_dataset("unrelated", {"display_name": "Existing catalog item"})

        source_counts, target_counts = migrate_sqlite_to_postgres(source, target, allow_existing=True)

        assert source_counts.datasets == 1
        assert target_counts.datasets == 2
        assert target.get_dataset("project-plan")["display_name"] == "Project plan"
        assert target.get_dataset("unrelated")["display_name"] == "Existing catalog item"
    finally:
        source.close()
        target.close()
