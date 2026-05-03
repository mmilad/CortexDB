"""SQLite-backed persistent store for CortexDB registry data.

Uses Python stdlib sqlite3 only — no extra dependencies.
WAL mode is enabled for safe concurrent reads under light write load.

Vectors are stored as JSON-encoded float arrays in TEXT columns.
This is intentional for the current embedded phase; sqlite-vec (ANN indexing)
will be added as an optional extension when installed, keeping the core
store dependency-free.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

_DB_PATH_ENV_VAR = "CORTEXDB_DB_PATH"
_DEFAULT_DB_PATH = "cortexdb.sqlite"

_DDL = """\
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_key        TEXT PRIMARY KEY,
    data               TEXT NOT NULL,
    embed_raw          TEXT,
    embedding          TEXT,
    embedding_model    TEXT,
    embedded_at        TEXT
);

CREATE TABLE IF NOT EXISTS tools (
    tool_key    TEXT PRIMARY KEY,
    data        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id          TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_key  TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_key  TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    join_fields TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_items (
    id              TEXT PRIMARY KEY,
    dataset_key     TEXT NOT NULL,
    raw_text        TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    embedding       TEXT,
    embedding_model TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (dataset_key) REFERENCES datasets(dataset_key)
);

CREATE INDEX IF NOT EXISTS idx_rel_source    ON relationships (source_type, source_key);
CREATE INDEX IF NOT EXISTS idx_rel_target    ON relationships (target_type, target_key);
CREATE INDEX IF NOT EXISTS idx_items_dataset ON memory_items (dataset_key);
"""

# Migration: add new columns to existing databases that predate them.
_MIGRATIONS = [
    "ALTER TABLE datasets ADD COLUMN embed_raw       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding_model TEXT",
    "ALTER TABLE datasets ADD COLUMN embedded_at     TEXT",
]


def _db_path() -> str:
    import os
    return os.environ.get(_DB_PATH_ENV_VAR, _DEFAULT_DB_PATH)


def _connect(path: str | None = None) -> sqlite3.Connection:
    p = path or _db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()
    # Run additive migrations idempotently (ignore "duplicate column" errors).
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


# ------------------------------------------------------------------
# Vector math (stdlib only; no numpy)
# ------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SqliteStore:
    """Thin wrapper around a single SQLite connection.

    Call `close()` on shutdown. Thread-safety: SQLite WAL + check_same_thread=False
    is safe for the single-process FastAPI workload this targets.
    """

    def __init__(self, path: str | None = None) -> None:
        self._conn = _connect(path)
        _init_db(self._conn)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def upsert_dataset(self, key: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO datasets (dataset_key, data) VALUES (?, ?)"
            " ON CONFLICT(dataset_key) DO UPDATE SET data = excluded.data",
            (key, json.dumps(data)),
        )
        self._conn.commit()

    def set_dataset_embedding(
        self,
        key: str,
        raw_text: str,
        embedding: list[float],
        model_id: str,
    ) -> None:
        self._conn.execute(
            """UPDATE datasets
               SET embed_raw = ?, embedding = ?, embedding_model = ?,
                   embedded_at = datetime('now')
               WHERE dataset_key = ?""",
            (raw_text, json.dumps(embedding), model_id, key),
        )
        self._conn.commit()

    def get_dataset(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def get_dataset_embedding(self, key: str) -> tuple[list[float] | None, str | None]:
        """Return (vector, model_id) or (None, None) if not yet embedded."""
        row = self._conn.execute(
            "SELECT embedding, embedding_model FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        if not row or row["embedding"] is None:
            return None, None
        return json.loads(row["embedding"]), row["embedding_model"]

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT dataset_key, data FROM datasets").fetchall()
        return {r["dataset_key"]: json.loads(r["data"]) for r in rows}

    def list_datasets_with_embeddings(self) -> list[dict[str, Any]]:
        """Return all datasets that have a stored embedding."""
        rows = self._conn.execute(
            "SELECT dataset_key, data, embedding, embedding_model "
            "FROM datasets WHERE embedding IS NOT NULL"
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "dataset_key": r["dataset_key"],
                "data": json.loads(r["data"]),
                "embedding": json.loads(r["embedding"]),
                "embedding_model": r["embedding_model"],
            })
        return result

    def delete_dataset(self, key: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM datasets WHERE dataset_key = ?", (key,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def upsert_tool(self, key: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO tools (tool_key, data) VALUES (?, ?)"
            " ON CONFLICT(tool_key) DO UPDATE SET data = excluded.data",
            (key, json.dumps(data)),
        )
        self._conn.commit()

    def get_tool(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM tools WHERE tool_key = ?", (key,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_tools(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT tool_key, data FROM tools").fetchall()
        return {r["tool_key"]: json.loads(r["data"]) for r in rows}

    def delete_tool(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM tools WHERE tool_key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def upsert_relationship(self, rel: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO relationships
               (id, source_type, source_key, target_type, target_key,
                edge_type, join_fields, description)
               VALUES (:id, :source_type, :source_key, :target_type, :target_key,
                       :edge_type, :join_fields, :description)
               ON CONFLICT(id) DO UPDATE SET
                 source_type = excluded.source_type,
                 source_key  = excluded.source_key,
                 target_type = excluded.target_type,
                 target_key  = excluded.target_key,
                 edge_type   = excluded.edge_type,
                 join_fields = excluded.join_fields,
                 description = excluded.description""",
            {
                "id": rel["id"],
                "source_type": rel["source_type"],
                "source_key": rel["source_key"],
                "target_type": rel["target_type"],
                "target_key": rel["target_key"],
                "edge_type": rel["edge_type"],
                "join_fields": json.dumps(rel.get("join_fields", [])),
                "description": rel.get("description", ""),
            },
        )
        self._conn.commit()

    def get_relationship(self, rel_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM relationships WHERE id = ?", (rel_id,)
        ).fetchone()
        return self._row_to_rel(row) if row else None

    def list_relationships(
        self,
        source_key: str | None = None,
        target_key: str | None = None,
    ) -> list[dict[str, Any]]:
        if source_key and target_key:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE source_key = ? OR target_key = ?",
                (source_key, target_key),
            ).fetchall()
        elif source_key:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE source_key = ? OR target_key = ?",
                (source_key, source_key),
            ).fetchall()
        elif target_key:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE target_key = ?", (target_key,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        return [self._row_to_rel(r) for r in rows]

    def delete_relationship(self, rel_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM relationships WHERE id = ?", (rel_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def adjacency(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        return [self._row_to_rel(r) for r in rows]

    @staticmethod
    def _row_to_rel(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["join_fields"] = json.loads(d["join_fields"])
        return d

    # ------------------------------------------------------------------
    # Memory items
    # ------------------------------------------------------------------

    def insert_memory_item(self, item: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO memory_items
               (id, dataset_key, raw_text, metadata, embedding, embedding_model)
               VALUES (:id, :dataset_key, :raw_text, :metadata, :embedding, :embedding_model)
               ON CONFLICT(id) DO UPDATE SET
                 raw_text        = excluded.raw_text,
                 metadata        = excluded.metadata,
                 embedding       = excluded.embedding,
                 embedding_model = excluded.embedding_model""",
            {
                "id": item["id"],
                "dataset_key": item["dataset_key"],
                "raw_text": item["raw_text"],
                "metadata": json.dumps(item.get("metadata", {})),
                "embedding": json.dumps(item["embedding"]) if item.get("embedding") else None,
                "embedding_model": item.get("embedding_model"),
            },
        )
        self._conn.commit()

    def list_memory_items(
        self,
        dataset_key: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM memory_items WHERE dataset_key = ? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (dataset_key, limit, offset),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_memory_item(self, item_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def delete_memory_item(self, item_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def search_memory_items(
        self,
        dataset_key: str,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over memory items.

        Loads all items for the dataset (with embeddings) into memory,
        computes cosine similarity in Python, returns top-k sorted by score.

        This is correct and fast for up to ~50k items. For larger corpora,
        replace with sqlite-vec ANN index (extension is optional and auto-detected).
        """
        rows = self._conn.execute(
            "SELECT * FROM memory_items WHERE dataset_key = ? AND embedding IS NOT NULL",
            (dataset_key,),
        ).fetchall()

        scored = []
        for row in rows:
            item = self._row_to_item(row)

            # Apply metadata filters if provided
            if metadata_filters:
                meta = item.get("metadata", {})
                if not all(meta.get(k) == v for k, v in metadata_filters.items()):
                    continue

            vec = item.pop("_embedding_raw", None)
            if vec is None:
                continue
            sim = cosine_similarity(query_vector, vec)
            item["score"] = round(sim, 6)
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count_memory_items(self, dataset_key: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM memory_items WHERE dataset_key = ?",
            (dataset_key,),
        ).fetchone()
        return row["n"] if row else 0

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        # Store raw embedding separately for search; strip from public dict
        raw_emb = d.pop("embedding", None)
        d["_embedding_raw"] = json.loads(raw_emb) if raw_emb else None
        return d


# ------------------------------------------------------------------
# Singleton + FastAPI dependency
# ------------------------------------------------------------------

_store: SqliteStore | None = None


def get_store() -> SqliteStore:
    global _store
    if _store is None:
        _store = SqliteStore()
    return _store


def close_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None
