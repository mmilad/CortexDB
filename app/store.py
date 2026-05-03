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
    is_deleted      INTEGER NOT NULL DEFAULT 0,
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
    "ALTER TABLE memory_items ADD COLUMN is_deleted  INTEGER NOT NULL DEFAULT 0",
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
        exists = self._conn.execute(
            "SELECT 1 FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        if not exists:
            return False
        # Hard-delete all memory items belonging to this dataset first.
        self._conn.execute("DELETE FROM memory_items WHERE dataset_key = ?", (key,))
        self._conn.execute("DELETE FROM datasets WHERE dataset_key = ?", (key,))
        self._conn.commit()
        return True

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
                 embedding_model = excluded.embedding_model,
                 is_deleted      = 0""",
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
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        base = "SELECT * FROM memory_items WHERE dataset_key = ?"
        if not include_deleted:
            base += " AND is_deleted = 0"
        base += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        rows = self._conn.execute(base, (dataset_key, limit, offset)).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_memory_item(self, item_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def soft_delete_memory_item(self, item_id: str) -> bool:
        cur = self._conn.execute(
            "UPDATE memory_items SET is_deleted = 1 WHERE id = ? AND is_deleted = 0",
            (item_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_memory_item(self, item_id: str) -> bool:
        """Hard delete — permanently removes the row."""
        cur = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def search_memory_items(
        self,
        dataset_key: str,
        query_vector: list[float] | None,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        keyword_query: str | None = None,
        vector_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search over memory items: vector + optional keyword blending.

        Scoring modes:
          vector only  (keyword_query=None):  score = cosine_similarity
          keyword only (query_vector=None):   score = 1.0 if keyword matches, else item excluded
          hybrid       (both provided):       score = vector_weight * vector_score
                                                    + (1 - vector_weight) * keyword_score

        Items without an embedding are included only in keyword-only mode.
        In hybrid mode items without an embedding receive vector_score=0.

        Fast for up to ~50k items (Python in-process). Replace with sqlite-vec
        ANN index for larger corpora.
        """
        require_embedding = query_vector is not None and keyword_query is None

        if require_embedding:
            rows = self._conn.execute(
                "SELECT * FROM memory_items "
                "WHERE dataset_key = ? AND embedding IS NOT NULL AND is_deleted = 0",
                (dataset_key,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_items WHERE dataset_key = ? AND is_deleted = 0",
                (dataset_key,),
            ).fetchall()

        kw_lower = keyword_query.lower() if keyword_query else None
        scored = []

        for row in rows:
            item = self._row_to_item(row)

            if metadata_filters:
                meta = item.get("metadata", {})
                if not all(meta.get(k) == v for k, v in metadata_filters.items()):
                    continue

            raw_vec = item.pop("_embedding_raw", None)

            # Compute keyword score
            kw_score: float | None = None
            if kw_lower is not None:
                kw_score = 1.0 if kw_lower in item.get("raw_text", "").lower() else 0.0
                if kw_score == 0.0 and query_vector is None:
                    continue  # keyword-only mode: exclude non-matching items

            # Compute vector score
            vec_score: float | None = None
            if query_vector is not None:
                if raw_vec is not None:
                    vec_score = round(cosine_similarity(query_vector, raw_vec), 6)
                else:
                    vec_score = 0.0

            # Blend
            if vec_score is not None and kw_score is not None:
                final = vector_weight * vec_score + (1.0 - vector_weight) * kw_score
            elif vec_score is not None:
                final = vec_score
            elif kw_score is not None:
                final = kw_score
            else:
                continue

            item["score"] = round(final, 6)
            item["vector_score"] = vec_score
            item["keyword_score"] = kw_score
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count_memory_items(self, dataset_key: str, include_deleted: bool = False) -> int:
        base = "SELECT COUNT(*) as n FROM memory_items WHERE dataset_key = ?"
        if not include_deleted:
            base += " AND is_deleted = 0"
        row = self._conn.execute(base, (dataset_key,)).fetchone()
        return row["n"] if row else 0

    def list_all_memory_items(self, dataset_key: str) -> list[dict[str, Any]]:
        """Return all non-deleted items for a dataset (including those without embeddings)."""
        rows = self._conn.execute(
            "SELECT * FROM memory_items WHERE dataset_key = ? AND is_deleted = 0 "
            "ORDER BY created_at ASC",
            (dataset_key,),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def update_memory_item_embedding(
        self,
        item_id: str,
        embedding: list[float],
        model_id: str,
    ) -> None:
        self._conn.execute(
            "UPDATE memory_items SET embedding = ?, embedding_model = ? WHERE id = ?",
            (json.dumps(embedding), model_id, item_id),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        # Store raw embedding separately for search; strip from public dict
        raw_emb = d.pop("embedding", None)
        d["_embedding_raw"] = json.loads(raw_emb) if raw_emb else None
        # Normalise is_deleted to bool
        d["is_deleted"] = bool(d.get("is_deleted", 0))
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
