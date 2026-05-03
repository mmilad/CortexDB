"""SQLite-backed persistent store for CortexDB registry data.

Uses Python stdlib sqlite3 only — no extra dependencies for basic operation.
WAL mode is enabled for safe concurrent reads under light write load.

Vector search strategy
----------------------
When the ``sqlite-vec`` package is installed (``pip install sqlite-vec``),
CortexDB builds one ``vec0`` virtual table per dataset as an ANN index.
Benchmarks show ~19 ms for KNN-10 over 20 000 rows at 768 dimensions.

When ``sqlite-vec`` is not installed, the store falls back to an in-process
Python cosine scan that is comfortable up to ~20 000 rows — matching our
target dataset size.

The JSON ``embedding`` column in ``memory_items`` is always the source of
truth.  The ``vec0`` tables are derived indices that can be rebuilt from it
at any time (e.g. via ``rebuild_vec_index``).

Vec table naming
----------------
A dataset with key ``"my-dataset"`` maps to a vec0 table named
``vec_items_my_dataset`` (non-alphanumeric chars replaced with ``_``).
The vec dimension for each dataset is stored in the ``vec_dim`` column of
the ``datasets`` table and is set the first time items with embeddings are
inserted.

Dimension changes during re-embed
----------------------------------
When ``re_embed_dataset`` changes the model and therefore the embedding
dimension, the caller should follow up with ``rebuild_vec_index(dataset_key)``
which drops the old vec0 table and recreates it at the new dimension.  The
re-embed API endpoint in ``app/api/memory.py`` does this automatically.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("cortexdb.store")

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

# Additive migrations — run idempotently on every startup.
_MIGRATIONS = [
    "ALTER TABLE datasets ADD COLUMN embed_raw       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding_model TEXT",
    "ALTER TABLE datasets ADD COLUMN embedded_at     TEXT",
    "ALTER TABLE memory_items ADD COLUMN is_deleted  INTEGER NOT NULL DEFAULT 0",
    # vec_dim tracks the embedding dimension of the vec0 ANN table for each dataset.
    "ALTER TABLE datasets ADD COLUMN vec_dim INTEGER",
]


# ------------------------------------------------------------------
# sqlite-vec optional extension
# ------------------------------------------------------------------

def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec into *conn*. Returns True on success."""
    try:
        sqlite_vec = importlib.import_module("sqlite_vec")
    except ModuleNotFoundError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as exc:
        logger.warning("sqlite-vec found but failed to load: %s", exc)
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False


def _vec_table_name(dataset_key: str) -> str:
    """Safe vec0 table name derived from a dataset key."""
    safe = re.sub(r"[^a-z0-9]", "_", dataset_key.lower())
    return f"vec_items_{safe}"


def _serialize_f32(vec: list[float]) -> bytes:
    """Convert a float list to sqlite-vec float32 binary format."""
    import struct
    return struct.pack(f"{len(vec)}f", *vec)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


# ------------------------------------------------------------------
# Vector math fallback (stdlib only; no numpy)
# ------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ------------------------------------------------------------------
# SqliteStore
# ------------------------------------------------------------------

class SqliteStore:
    """Thin wrapper around a single SQLite connection.

    Call ``close()`` on shutdown. Thread-safety: SQLite WAL +
    ``check_same_thread=False`` is safe for the single-process FastAPI
    workload this targets.

    Vector search uses sqlite-vec ANN when available, falling back to
    Python cosine scan otherwise.
    """

    def __init__(self, path: str | None = None) -> None:
        self._conn = _connect(path)
        _init_db(self._conn)
        self._vec_enabled = _try_load_sqlite_vec(self._conn)
        if self._vec_enabled:
            logger.info("sqlite-vec loaded — ANN vector search enabled.")
        else:
            logger.info(
                "sqlite-vec not available — falling back to Python cosine scan. "
                "Install 'sqlite-vec' for faster vector search."
            )

    def close(self) -> None:
        self._conn.close()

    @property
    def vec_enabled(self) -> bool:
        return self._vec_enabled

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
        self._conn.execute("DELETE FROM memory_items WHERE dataset_key = ?", (key,))
        self._conn.execute(
            "DELETE FROM relationships WHERE source_key = ? OR target_key = ?", (key, key)
        )
        self._conn.execute("DELETE FROM datasets WHERE dataset_key = ?", (key,))
        # Drop the vec0 ANN table if it exists.
        if self._vec_enabled:
            tbl = _vec_table_name(key)
            self._conn.execute(f"DROP TABLE IF EXISTS {tbl}")
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
        exists = self._conn.execute(
            "SELECT 1 FROM tools WHERE tool_key = ?", (key,)
        ).fetchone()
        if not exists:
            return False
        self._conn.execute(
            "DELETE FROM relationships WHERE source_key = ? OR target_key = ?", (key, key)
        )
        self._conn.execute("DELETE FROM tools WHERE tool_key = ?", (key,))
        self._conn.commit()
        return True

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
    # Vec0 index helpers (only called when _vec_enabled)
    # ------------------------------------------------------------------

    def _vec_table_exists(self, dataset_key: str) -> bool:
        tbl = _vec_table_name(dataset_key)
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
        ).fetchone()
        return row is not None

    def _create_vec_table(self, dataset_key: str, dim: int) -> None:
        """Create the vec0 table for a dataset at a given dimension."""
        tbl = _vec_table_name(dataset_key)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {tbl} "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
        self._conn.execute(
            "UPDATE datasets SET vec_dim = ? WHERE dataset_key = ?", (dim, dataset_key)
        )
        self._conn.commit()

    def _drop_vec_table(self, dataset_key: str) -> None:
        tbl = _vec_table_name(dataset_key)
        self._conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        self._conn.execute(
            "UPDATE datasets SET vec_dim = NULL WHERE dataset_key = ?", (dataset_key,)
        )
        self._conn.commit()

    def _get_vec_dim(self, dataset_key: str) -> int | None:
        """Return the stored vec_dim for the dataset, or None if not set."""
        row = self._conn.execute(
            "SELECT vec_dim FROM datasets WHERE dataset_key = ?", (dataset_key,)
        ).fetchone()
        return row["vec_dim"] if row else None

    def _ensure_vec_table(self, dataset_key: str, dim: int) -> bool:
        """Ensure a vec0 table exists for the dataset at the given dimension.

        Returns True if the table is ready at the requested dimension.
        Returns False if the table exists at a *different* dimension
        (caller should call rebuild_vec_index to reconcile).
        """
        stored_dim = self._get_vec_dim(dataset_key)
        if stored_dim is None:
            self._create_vec_table(dataset_key, dim)
            return True
        if stored_dim != dim:
            return False
        if not self._vec_table_exists(dataset_key):
            # vec_dim is set but table is missing (e.g. after a manual DROP)
            self._create_vec_table(dataset_key, dim)
        return True

    def _vec_upsert(self, dataset_key: str, rowid: int, embedding: list[float]) -> None:
        """Insert or replace a vector in the vec0 table (vec0 uses DELETE+INSERT)."""
        tbl = _vec_table_name(dataset_key)
        self._conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (rowid,))
        self._conn.execute(
            f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_f32(embedding)),
        )

    def _vec_delete(self, dataset_key: str, rowid: int) -> None:
        if self._vec_table_exists(dataset_key):
            tbl = _vec_table_name(dataset_key)
            self._conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (rowid,))

    def _rowid_for_item(self, item_id: str) -> int | None:
        """Return the SQLite rowid of a memory_items row by its TEXT id."""
        row = self._conn.execute(
            "SELECT rowid FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return row[0] if row else None

    def rebuild_vec_index(self, dataset_key: str) -> int:
        """Drop and rebuild the vec0 table for *dataset_key* from stored embeddings.

        Call this after a re-embed operation that may have changed the
        embedding dimension.  Returns the number of rows indexed.
        """
        if not self._vec_enabled:
            return 0

        rows = self._conn.execute(
            "SELECT rowid, embedding FROM memory_items "
            "WHERE dataset_key = ? AND embedding IS NOT NULL AND is_deleted = 0",
            (dataset_key,),
        ).fetchall()

        if not rows:
            self._drop_vec_table(dataset_key)
            return 0

        # Determine dimension from the first row
        first_vec = json.loads(rows[0]["embedding"])
        dim = len(first_vec)

        self._drop_vec_table(dataset_key)
        self._create_vec_table(dataset_key, dim)

        tbl = _vec_table_name(dataset_key)
        self._conn.execute("BEGIN")
        for row in rows:
            vec = json.loads(row["embedding"])
            if len(vec) != dim:
                continue  # skip dimension-mismatched rows (shouldn't happen)
            self._conn.execute(
                f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
                (row["rowid"], _serialize_f32(vec)),
            )
        self._conn.execute("COMMIT")
        return len(rows)

    # ------------------------------------------------------------------
    # Memory items
    # ------------------------------------------------------------------

    def insert_memory_item(self, item: dict[str, Any]) -> None:
        embedding = item.get("embedding")
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
                "embedding": json.dumps(embedding) if embedding else None,
                "embedding_model": item.get("embedding_model"),
            },
        )
        self._conn.commit()

        # Sync to vec0 index when embedding is provided and vec is enabled.
        if self._vec_enabled and embedding:
            dim = len(embedding)
            ready = self._ensure_vec_table(item["dataset_key"], dim)
            if ready:
                rowid = self._rowid_for_item(item["id"])
                if rowid is not None:
                    self._vec_upsert(item["dataset_key"], rowid, embedding)
                    self._conn.commit()
            else:
                logger.warning(
                    "Dataset '%s' vec0 table has a different dimension; "
                    "call rebuild_vec_index to reconcile.",
                    item["dataset_key"],
                )

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
        """Hard delete — permanently removes the row and its vec0 entry."""
        # Need rowid and dataset_key before deletion for vec cleanup.
        row = self._conn.execute(
            "SELECT rowid, dataset_key FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        cur = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()
        if cur.rowcount > 0 and row and self._vec_enabled:
            self._vec_delete(row["dataset_key"], row["rowid"])
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
          keyword only (query_vector=None):   score = BM25 keyword score
          hybrid       (both provided):       score = vector_weight * vector_score
                                                    + (1 - vector_weight) * bm25_score

        When sqlite-vec is available, vector search uses the ANN index
        (fast even at 20 000 rows).  When unavailable, falls back to a
        Python cosine scan.

        Items without an embedding are included only in keyword-only mode.
        In hybrid mode, items without an embedding receive vector_score=0.
        """
        needs_vector = query_vector is not None

        if needs_vector and keyword_query is None:
            # Vector-only: use ANN path when available.
            return self._search_vector_only(
                dataset_key, query_vector, top_k, metadata_filters  # type: ignore[arg-type]
            )

        if query_vector is None and keyword_query is not None:
            # Keyword-only: BM25 over all non-deleted items.
            return self._search_keyword_only(dataset_key, keyword_query, top_k, metadata_filters)

        if needs_vector and keyword_query is not None:
            # Hybrid: merge ANN results with BM25.
            return self._search_hybrid(
                dataset_key, query_vector, keyword_query,  # type: ignore[arg-type]
                top_k, metadata_filters, vector_weight
            )

        return []

    # ------ internal search implementations ------

    def _load_items_for_scan(
        self,
        dataset_key: str,
        require_embedding: bool,
    ) -> list[dict[str, Any]]:
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
        return [self._row_to_item(r) for r in rows]

    def _apply_metadata_filter(
        self,
        items: list[dict[str, Any]],
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not metadata_filters:
            return items
        return [
            it for it in items
            if all(it.get("metadata", {}).get(k) == v for k, v in metadata_filters.items())
        ]

    def _search_vector_only(
        self,
        dataset_key: str,
        query_vector: list[float],
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self._vec_enabled and self._vec_table_exists(dataset_key):
            return self._vec_knn_search(
                dataset_key, query_vector, top_k, metadata_filters,
                keyword_query=None, vector_weight=1.0
            )
        # Fallback: Python cosine scan
        items = self._load_items_for_scan(dataset_key, require_embedding=True)
        items = self._apply_metadata_filter(items, metadata_filters)
        scored = []
        for item in items:
            raw_vec = item.pop("_embedding_raw", None)
            if raw_vec is None:
                continue
            score = round(cosine_similarity(query_vector, raw_vec), 6)
            item["score"] = score
            item["vector_score"] = score
            item["keyword_score"] = None
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _search_keyword_only(
        self,
        dataset_key: str,
        keyword_query: str,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        items = self._load_items_for_scan(dataset_key, require_embedding=False)
        items = self._apply_metadata_filter(items, metadata_filters)
        for item in items:
            item.pop("_embedding_raw", None)
        scored = _bm25_score(items, keyword_query)
        scored = [it for it in scored if it["keyword_score"] > 0]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _search_hybrid(
        self,
        dataset_key: str,
        query_vector: list[float],
        keyword_query: str,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
        vector_weight: float,
    ) -> list[dict[str, Any]]:
        if self._vec_enabled and self._vec_table_exists(dataset_key):
            return self._vec_knn_search(
                dataset_key, query_vector, top_k, metadata_filters,
                keyword_query=keyword_query, vector_weight=vector_weight
            )
        # Fallback: Python scan + BM25 blend
        items = self._load_items_for_scan(dataset_key, require_embedding=False)
        items = self._apply_metadata_filter(items, metadata_filters)

        # BM25 scores over full candidate set
        items_with_kw = _bm25_score(items, keyword_query)

        scored = []
        for item in items_with_kw:
            raw_vec = item.pop("_embedding_raw", None)
            vec_score = round(cosine_similarity(query_vector, raw_vec), 6) if raw_vec else 0.0
            kw_score = item.get("keyword_score", 0.0)
            final = vector_weight * vec_score + (1.0 - vector_weight) * kw_score
            item["score"] = round(final, 6)
            item["vector_score"] = vec_score
            item["keyword_score"] = kw_score
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _vec_knn_search(
        self,
        dataset_key: str,
        query_vector: list[float],
        top_k: int,
        metadata_filters: dict[str, Any] | None,
        keyword_query: str | None,
        vector_weight: float,
    ) -> list[dict[str, Any]]:
        """ANN search via vec0 with optional BM25 blending and metadata filtering.

        Fetches a candidate set of min(total, top_k * 20) items from the ANN
        index, then applies metadata filtering and BM25 blending in Python.
        At <=20 000 rows the candidate set is the full dataset, which is fine.
        """
        tbl = _vec_table_name(dataset_key)
        total = self.count_memory_items(dataset_key)
        candidate_k = min(total, max(top_k * 20, top_k))

        # ANN KNN — returns (rowid, distance) sorted by distance (ascending = more similar)
        knn_rows = self._conn.execute(
            f"SELECT rowid, distance FROM {tbl} "
            f"WHERE embedding MATCH ? AND k = ?",
            (_serialize_f32(query_vector), candidate_k),
        ).fetchall()

        if not knn_rows:
            return []

        rowid_to_dist = {r[0]: r[1] for r in knn_rows}
        placeholders = ",".join("?" for _ in rowid_to_dist)
        items_rows = self._conn.execute(
            f"SELECT *, rowid FROM memory_items "
            f"WHERE rowid IN ({placeholders}) AND is_deleted = 0",
            list(rowid_to_dist.keys()),
        ).fetchall()

        items = []
        for row in items_rows:
            item = self._row_to_item(row)
            item["_rowid"] = dict(row)["rowid"]
            items.append(item)

        items = self._apply_metadata_filter(items, metadata_filters)

        if keyword_query:
            items = _bm25_score(items, keyword_query)

        scored = []
        for item in items:
            item.pop("_embedding_raw", None)
            rowid = item.pop("_rowid", None)
            ann_dist = rowid_to_dist.get(rowid, 1.0)
            # Convert cosine distance (0=identical, 2=opposite) to similarity [0,1]
            vec_score = round(max(0.0, 1.0 - ann_dist), 6)

            if keyword_query:
                kw_score = item.get("keyword_score", 0.0)
                final = vector_weight * vec_score + (1.0 - vector_weight) * kw_score
            else:
                kw_score = None
                final = vec_score

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

        # Sync to vec0 index.
        if self._vec_enabled:
            row = self._conn.execute(
                "SELECT rowid, dataset_key FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row:
                dataset_key = row["dataset_key"]
                rowid = row["rowid"]
                dim = len(embedding)
                ready = self._ensure_vec_table(dataset_key, dim)
                if ready:
                    self._vec_upsert(dataset_key, rowid, embedding)
                    self._conn.commit()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        raw_emb = d.pop("embedding", None)
        d["_embedding_raw"] = json.loads(raw_emb) if raw_emb else None
        d["is_deleted"] = bool(d.get("is_deleted", 0))
        # rowid is a pseudo-column; strip it from the public dict
        d.pop("rowid", None)
        return d


# ------------------------------------------------------------------
# BM25 keyword scorer (stdlib only)
# ------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser — splits on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(
    items: list[dict[str, Any]],
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Compute BM25 scores for *items* against *query*.

    Adds ``keyword_score`` and sets ``score = keyword_score`` on each item.
    Items with zero score are retained (caller filters if needed).

    BM25 parameters: k1=1.5 (term-frequency saturation), b=0.75 (length norm).
    """
    if not items:
        return items

    query_terms = set(_tokenize(query))
    if not query_terms:
        for it in items:
            it["keyword_score"] = 0.0
            it["score"] = 0.0
        return items

    # Tokenize all documents
    doc_tokens: list[list[str]] = [_tokenize(it.get("raw_text", "")) for it in items]
    N = len(items)
    avg_dl = sum(len(t) for t in doc_tokens) / N if N else 1.0

    # IDF per query term: log((N - df + 0.5) / (df + 0.5) + 1)
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in query_terms:
            if term in tokens:
                df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term in query_terms:
        n_t = df.get(term, 0)
        idf[term] = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    # Score each document
    for item, tokens in zip(items, doc_tokens):
        dl = len(tokens)
        tf_counts: dict[str, int] = {}
        for t in tokens:
            if t in query_terms:
                tf_counts[t] = tf_counts.get(t, 0) + 1

        bm25 = 0.0
        for term in query_terms:
            tf = tf_counts.get(term, 0)
            if tf == 0:
                continue
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            bm25 += idf[term] * norm_tf

        # Normalise to [0, 1] using a soft-max approach:
        # max possible BM25 score per term ≈ idf * (k1+1)
        max_possible = sum(idf[t] * (k1 + 1) for t in query_terms) or 1.0
        norm_score = round(min(bm25 / max_possible, 1.0), 6)
        item["keyword_score"] = norm_score
        item["score"] = norm_score

    return items


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
