"""Idempotent migration from a CortexDB SQLite file to PostgreSQL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from app.store import PostgresStore, SqliteStore


@dataclass(frozen=True)
class MigrationCounts:
    datasets: int
    tools: int
    rule_packs: int
    relationships: int
    sessions: int
    raw_texts: int
    messages: int
    summaries: int
    memory_items: int


def _counts(store: Any) -> MigrationCounts:
    sessions = store.list_sessions()
    messages = sum(len(store.list_session_messages(session["id"])) for session in sessions)
    summaries = sum(len(store.list_session_summaries(session["id"])) for session in sessions)
    raw_texts = len(store.list_raw_texts())
    memory_items = sum(
        len(store.list_memory_items(dataset_key, include_deleted=True))
        for dataset_key in store.list_datasets()
    )
    return MigrationCounts(
        datasets=len(store.list_datasets()),
        tools=len(store.list_tools()),
        rule_packs=len(store.list_ingest_rule_packs()),
        relationships=len(store.list_relationships()),
        sessions=len(sessions),
        raw_texts=raw_texts,
        messages=messages,
        summaries=summaries,
        memory_items=memory_items,
    )


def migrate_sqlite_to_postgres(
    source: SqliteStore,
    target: PostgresStore,
    *,
    allow_existing: bool = False,
) -> tuple[MigrationCounts, MigrationCounts]:
    """Copy all durable SQLite records without changing their stable IDs.

    In strict mode, the target is expected to be an empty/equivalent database.
    In merge mode, existing target records are preserved and only records with
    keys that are not already present are imported. No source rows are deleted
    or modified, and merge mode never overwrites shared target state.
    """
    existing_dataset_keys = set(target.list_datasets()) if allow_existing else set()
    for key, data in source.list_datasets().items():
        if not allow_existing or target.get_dataset(key) is None:
            target.upsert_dataset(key, data)
    for row in source.list_datasets_with_embeddings():
        if not allow_existing or row["dataset_key"] not in existing_dataset_keys:
            target.set_dataset_embedding(row["dataset_key"], row.get("embed_raw") or "", row["embedding"], row["embedding_model"])

    for key, data in source.list_tools().items():
        if not allow_existing or target.get_tool(key) is None:
            target.upsert_tool(key, data)
    for rule in source.list_ingest_rule_packs():
        if not allow_existing or target.get_ingest_rule_pack(rule["key"], rule.get("namespace")) is None:
            target.upsert_ingest_rule_pack(rule["key"], rule, rule.get("namespace"))
    for relationship in source.list_relationships():
        if not allow_existing or target.get_relationship(relationship["id"]) is None:
            target.upsert_relationship(relationship)

    for session in source.list_sessions():
        if not allow_existing or target.get_session(session["id"]) is None:
            target.upsert_session(session)
    for raw_text in source.list_raw_texts():
        if not allow_existing or target.get_raw_text(raw_text["id"]) is None:
            target.insert_raw_text(raw_text)
    for session in source.list_sessions():
        for summary in source.list_session_summaries(session["id"]):
            existing_summaries = {item["id"] for item in target.list_session_summaries(session["id"])}
            if not allow_existing or summary["id"] not in existing_summaries:
                target.insert_session_summary(summary)
        for message in source.list_session_messages(session["id"]):
            existing_messages = {item["id"] for item in target.list_session_messages(session["id"])}
            if not allow_existing or message["id"] not in existing_messages:
                target.insert_session_message(message)

    for dataset_key in source.list_datasets():
        for item in source.list_memory_items(dataset_key, include_deleted=True):
            if not allow_existing or target.get_memory_item(item["id"]) is None:
                target.insert_memory_item(item)

    source_counts = _counts(source)
    target_counts = _counts(target)
    counts_match = (
        all(
            getattr(source_counts, field) == getattr(target_counts, field)
            for field in MigrationCounts.__dataclass_fields__
        )
        if not allow_existing
        else all(
            getattr(source_counts, field) <= getattr(target_counts, field)
            for field in MigrationCounts.__dataclass_fields__
        )
    )
    if not counts_match:
        raise RuntimeError(
            "SQLite to PostgreSQL migration count mismatch" + (" in merge mode" if allow_existing else "") + ":\n"
            f"source={json.dumps(asdict(source_counts), sort_keys=True)}\n"
            f"target={json.dumps(asdict(target_counts), sort_keys=True)}"
        )
    return source_counts, target_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a CortexDB SQLite database into PostgreSQL.")
    parser.add_argument("--source", default="cortexdb.sqlite", help="SQLite database path.")
    parser.add_argument("--database-url", help="PostgreSQL URL; defaults to CORTEXDB_DATABASE_URL.")
    parser.add_argument("--schema", default="cortexdb", help="PostgreSQL schema name.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Preserve pre-existing target records and import only missing source records.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"SQLite source does not exist: {source_path}")

    source = SqliteStore(str(source_path))
    target = PostgresStore(args.database_url, schema=args.schema)
    try:
        source_counts, target_counts = migrate_sqlite_to_postgres(source, target, allow_existing=args.merge)
    finally:
        source.close()
        target.close()

    print(json.dumps({"source": asdict(source_counts), "target": asdict(target_counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
