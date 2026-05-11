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

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.store import vec as vec_mod
from app.store.search import bm25_score, cosine_similarity

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
    embedded_at        TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tools (
    tool_key    TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_rule_packs (
    key        TEXT NOT NULL,
    namespace  TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'active',
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (namespace, key)
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

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    type           TEXT NOT NULL DEFAULT 'chat',
    scope_mode     TEXT NOT NULL DEFAULT 'namespace',
    namespace      TEXT,
    dataset_policy TEXT NOT NULL DEFAULT 'create_if_needed',
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_texts (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'unknown',
    relations       TEXT NOT NULL DEFAULT '{}',
    score           REAL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    embedding       TEXT,
    embedding_model TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    summary       TEXT NOT NULL,
    message_ids   TEXT NOT NULL DEFAULT '[]',
    token_estimate INTEGER NOT NULL DEFAULT 0,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_messages (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    role                TEXT NOT NULL,
    content             TEXT NOT NULL,
    raw_text_id         TEXT,
    token_estimate      INTEGER NOT NULL DEFAULT 0,
    autocontext_enabled INTEGER NOT NULL DEFAULT 1,
    summary_id          TEXT,
    metadata            TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (raw_text_id) REFERENCES raw_texts(id),
    FOREIGN KEY (summary_id) REFERENCES session_summaries(id)
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
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (dataset_key) REFERENCES datasets(dataset_key)
);

CREATE INDEX IF NOT EXISTS idx_rel_source    ON relationships (source_type, source_key);
CREATE INDEX IF NOT EXISTS idx_rel_target    ON relationships (target_type, target_key);
CREATE INDEX IF NOT EXISTS idx_items_dataset ON memory_items (dataset_key);
CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_raw_texts_created ON raw_texts (created_at);
CREATE INDEX IF NOT EXISTS idx_session_summaries_session ON session_summaries (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ingest_rule_packs_status ON ingest_rule_packs (namespace, status);
"""

# Additive migrations — run idempotently on every startup.
_MIGRATIONS = [
    "ALTER TABLE datasets ADD COLUMN embed_raw       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding       TEXT",
    "ALTER TABLE datasets ADD COLUMN embedding_model TEXT",
    "ALTER TABLE datasets ADD COLUMN embedded_at     TEXT",
    "ALTER TABLE memory_items ADD COLUMN is_deleted  INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE datasets ADD COLUMN vec_dim INTEGER",
    "ALTER TABLE datasets ADD COLUMN created_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
    "ALTER TABLE datasets ADD COLUMN updated_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
    "ALTER TABLE tools ADD COLUMN created_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
    "ALTER TABLE tools ADD COLUMN updated_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
    "ALTER TABLE memory_items ADD COLUMN updated_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
]


def _db_path() -> str:
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
        self._vec_enabled = vec_mod.try_load_sqlite_vec(self._conn)
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
            " ON CONFLICT(dataset_key) DO UPDATE SET data = excluded.data, updated_at = datetime('now')",
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
                   embedded_at = datetime('now'), updated_at = datetime('now')
               WHERE dataset_key = ?""",
            (raw_text, json.dumps(embedding), model_id, key),
        )
        self._conn.commit()

    def get_dataset(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data, created_at, updated_at FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["data"])
        data.setdefault("created_at", row["created_at"])
        data.setdefault("updated_at", row["updated_at"])
        return data

    def get_dataset_embedding(self, key: str) -> tuple[list[float] | None, str | None]:
        """Return (vector, model_id) or (None, None) if not yet embedded."""
        row = self._conn.execute(
            "SELECT embedding, embedding_model FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        if not row or row["embedding"] is None:
            return None, None
        return json.loads(row["embedding"]), row["embedding_model"]

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT dataset_key, data, created_at, updated_at FROM datasets").fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = json.loads(row["data"])
            data.setdefault("created_at", row["created_at"])
            data.setdefault("updated_at", row["updated_at"])
            result[row["dataset_key"]] = data
        return result

    def list_datasets_with_embeddings(self) -> list[dict[str, Any]]:
        """Return all datasets that have a stored embedding."""
        rows = self._conn.execute(
            "SELECT dataset_key, data, embedding, embedding_model, created_at, updated_at "
            "FROM datasets WHERE embedding IS NOT NULL"
        ).fetchall()
        result = []
        for r in rows:
            data = json.loads(r["data"])
            data.setdefault("created_at", r["created_at"])
            data.setdefault("updated_at", r["updated_at"])
            result.append({
                "dataset_key": r["dataset_key"],
                "data": data,
                "embedding": json.loads(r["embedding"]),
                "embedding_model": r["embedding_model"],
            })
        return result

    def delete_dataset(self, key: str) -> bool:
        """Delete a dataset and cascade to its memory items, their relationships,
        dataset-level relationships, and the vec0 ANN table."""
        exists = self._conn.execute(
            "SELECT 1 FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        if not exists:
            return False

        # Collect all item ids before deleting them so we can purge their edges.
        item_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT id FROM memory_items WHERE dataset_key = ?", (key,)
            ).fetchall()
        ]

        # Cascade: relationships that reference any of the items.
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            self._conn.execute(
                f"DELETE FROM relationships WHERE source_key IN ({placeholders})"
                f" OR target_key IN ({placeholders})",
                item_ids + item_ids,
            )

        self._conn.execute("DELETE FROM memory_items WHERE dataset_key = ?", (key,))

        # Dataset-level relationships (edges whose key is the dataset key itself).
        self._conn.execute(
            "DELETE FROM relationships WHERE source_key = ? OR target_key = ?", (key, key)
        )
        self._conn.execute("DELETE FROM datasets WHERE dataset_key = ?", (key,))

        if self._vec_enabled:
            vec_mod.drop_table(self._conn, key)

        self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def upsert_tool(self, key: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO tools (tool_key, data) VALUES (?, ?)"
            " ON CONFLICT(tool_key) DO UPDATE SET data = excluded.data, updated_at = datetime('now')",
            (key, json.dumps(data)),
        )
        self._conn.commit()

    def get_tool(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data, created_at, updated_at FROM tools WHERE tool_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["data"])
        data.setdefault("created_at", row["created_at"])
        data.setdefault("updated_at", row["updated_at"])
        return data

    def list_tools(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT tool_key, data, created_at, updated_at FROM tools").fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = json.loads(row["data"])
            data.setdefault("created_at", row["created_at"])
            data.setdefault("updated_at", row["updated_at"])
            result[row["tool_key"]] = data
        return result

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
    # Ingest rule packs
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace_key(namespace: str | None) -> str:
        return namespace or ""

    def upsert_ingest_rule_pack(self, key: str, data: dict[str, Any], namespace: str | None = None) -> None:
        ns = self._namespace_key(namespace)
        status = data.get("status", "active")
        self._conn.execute(
            """INSERT INTO ingest_rule_packs (key, namespace, status, data)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET
                 status = excluded.status,
                 data = excluded.data,
                 updated_at = datetime('now')""",
            (key, ns, status, json.dumps(data)),
        )
        self._conn.commit()

    def get_ingest_rule_pack(self, key: str, namespace: str | None = None) -> dict[str, Any] | None:
        ns = self._namespace_key(namespace)
        row = self._conn.execute(
            "SELECT key, namespace, data, created_at, updated_at FROM ingest_rule_packs "
            "WHERE namespace = ? AND key = ?",
            (ns, key),
        ).fetchone()
        return self._row_to_ingest_rule_pack(row) if row else None

    def list_ingest_rule_packs(
        self,
        namespace: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        ns = self._namespace_key(namespace)
        if active_only:
            rows = self._conn.execute(
                "SELECT key, namespace, data, created_at, updated_at FROM ingest_rule_packs "
                "WHERE namespace = ? AND status = 'active' ORDER BY key",
                (ns,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, namespace, data, created_at, updated_at FROM ingest_rule_packs "
                "WHERE namespace = ? ORDER BY key",
                (ns,),
            ).fetchall()
        return [self._row_to_ingest_rule_pack(row) for row in rows]

    def delete_ingest_rule_pack(self, key: str, namespace: str | None = None) -> bool:
        ns = self._namespace_key(namespace)
        cur = self._conn.execute(
            "DELETE FROM ingest_rule_packs WHERE namespace = ? AND key = ?",
            (ns, key),
        )
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_ingest_rule_pack(row: sqlite3.Row) -> dict[str, Any]:
        data = json.loads(row["data"])
        data.setdefault("key", row["key"])
        data["namespace"] = data.get("namespace") or row["namespace"] or None
        data["created_at"] = data.get("created_at") or row["created_at"]
        data["updated_at"] = data.get("updated_at") or row["updated_at"]
        return data

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
    # Sessions and raw text audit
    # ------------------------------------------------------------------

    def upsert_session(self, session: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO sessions
               (id, type, scope_mode, namespace, dataset_policy, metadata)
               VALUES (:id, :type, :scope_mode, :namespace, :dataset_policy, :metadata)
               ON CONFLICT(id) DO UPDATE SET
                 type = excluded.type,
                 scope_mode = excluded.scope_mode,
                 namespace = excluded.namespace,
                 dataset_policy = excluded.dataset_policy,
                 metadata = excluded.metadata,
                 updated_at = datetime('now')""",
            {
                "id": session["id"],
                "type": session.get("type", "chat"),
                "scope_mode": session.get("scope_mode", "namespace"),
                "namespace": session.get("namespace"),
                "dataset_policy": session.get("dataset_policy", "create_if_needed"),
                "metadata": json.dumps(session.get("metadata", {})),
            },
        )
        self._conn.commit()

    def ensure_session(
        self,
        session_id: str = "main",
        *,
        type: str = "chat",
        scope_mode: str = "namespace",
        namespace: str | None = None,
        dataset_policy: str = "create_if_needed",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.get_session(session_id)
        if existing:
            return existing
        self.upsert_session({
            "id": session_id,
            "type": type,
            "scope_mode": scope_mode,
            "namespace": namespace,
            "dataset_policy": dataset_policy,
            "metadata": metadata or {},
        })
        created = self.get_session(session_id)
        if created is None:
            raise RuntimeError(f"could not create session '{session_id}'")
        return created

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [self._row_to_session(r) for r in rows]

    def insert_raw_text(self, raw: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO raw_texts
               (id, text, source, relations, score, metadata, embedding, embedding_model)
               VALUES (:id, :text, :source, :relations, :score, :metadata, :embedding, :embedding_model)
               ON CONFLICT(id) DO UPDATE SET
                 text = excluded.text,
                 source = excluded.source,
                 relations = excluded.relations,
                 score = excluded.score,
                 metadata = excluded.metadata,
                 embedding = excluded.embedding,
                 embedding_model = excluded.embedding_model""",
            {
                "id": raw["id"],
                "text": raw["text"],
                "source": raw.get("source", "unknown"),
                "relations": json.dumps(raw.get("relations", {})),
                "score": raw.get("score"),
                "metadata": json.dumps(raw.get("metadata", {})),
                "embedding": json.dumps(raw.get("embedding")) if raw.get("embedding") else None,
                "embedding_model": raw.get("embedding_model"),
            },
        )
        self._conn.commit()

    def get_raw_text(self, raw_text_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM raw_texts WHERE id = ?", (raw_text_id,)).fetchone()
        return self._row_to_raw_text(row) if row else None

    def insert_session_message(self, message: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO session_messages
               (id, session_id, role, content, raw_text_id, token_estimate,
                autocontext_enabled, summary_id, metadata)
               VALUES (:id, :session_id, :role, :content, :raw_text_id, :token_estimate,
                       :autocontext_enabled, :summary_id, :metadata)
               ON CONFLICT(id) DO UPDATE SET
                 role = excluded.role,
                 content = excluded.content,
                 raw_text_id = excluded.raw_text_id,
                 token_estimate = excluded.token_estimate,
                 autocontext_enabled = excluded.autocontext_enabled,
                 summary_id = excluded.summary_id,
                 metadata = excluded.metadata""",
            {
                "id": message["id"],
                "session_id": message["session_id"],
                "role": message.get("role", "user"),
                "content": message["content"],
                "raw_text_id": message.get("raw_text_id"),
                "token_estimate": message.get("token_estimate", 0),
                "autocontext_enabled": 1 if message.get("autocontext_enabled", True) else 0,
                "summary_id": message.get("summary_id"),
                "metadata": json.dumps(message.get("metadata", {})),
            },
        )
        self._conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (message["session_id"],))
        self._conn.commit()

    def list_session_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        autocontext_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM session_messages WHERE session_id = ?"
        args: list[Any] = [session_id]
        if autocontext_only:
            query += " AND autocontext_enabled = 1"
        query += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        rows = self._conn.execute(query, args).fetchall()
        return [self._row_to_session_message(r) for r in rows]

    def insert_session_summary(self, summary: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO session_summaries
               (id, session_id, summary, message_ids, token_estimate, metadata)
               VALUES (:id, :session_id, :summary, :message_ids, :token_estimate, :metadata)
               ON CONFLICT(id) DO UPDATE SET
                 summary = excluded.summary,
                 message_ids = excluded.message_ids,
                 token_estimate = excluded.token_estimate,
                 metadata = excluded.metadata""",
            {
                "id": summary["id"],
                "session_id": summary["session_id"],
                "summary": summary["summary"],
                "message_ids": json.dumps(summary.get("message_ids", [])),
                "token_estimate": summary.get("token_estimate", 0),
                "metadata": json.dumps(summary.get("metadata", {})),
            },
        )
        self._conn.commit()

    def list_session_summaries(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM session_summaries WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_session_summary(r) for r in rows]

    def disable_session_messages_for_autocontext(
        self,
        session_id: str,
        message_ids: list[str],
        summary_id: str,
    ) -> int:
        if not message_ids:
            return 0
        placeholders = ",".join("?" for _ in message_ids)
        cur = self._conn.execute(
            f"""UPDATE session_messages
                SET autocontext_enabled = 0, summary_id = ?
                WHERE session_id = ? AND id IN ({placeholders})""",
            [summary_id, session_id, *message_ids],
        )
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        return d

    @staticmethod
    def _row_to_raw_text(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["relations"] = json.loads(d["relations"])
        d["metadata"] = json.loads(d["metadata"])
        raw_emb = d.pop("embedding", None)
        d["_embedding_raw"] = json.loads(raw_emb) if raw_emb else None
        return d

    @staticmethod
    def _row_to_session_message(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        d["autocontext_enabled"] = bool(d.get("autocontext_enabled", 1))
        return d

    @staticmethod
    def _row_to_session_summary(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["message_ids"] = json.loads(d["message_ids"])
        d["metadata"] = json.loads(d["metadata"])
        return d

    # ------------------------------------------------------------------
    # Vec0 index helpers (only called when _vec_enabled)
    # ------------------------------------------------------------------

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
            vec_mod.drop_table(self._conn, dataset_key)
            return 0

        first_vec = json.loads(rows[0]["embedding"])
        dim = len(first_vec)

        vec_mod.drop_table(self._conn, dataset_key)
        vec_mod.create_table(self._conn, dataset_key, dim)

        tbl = vec_mod.vec_table_name(dataset_key)
        self._conn.execute("BEGIN")
        for row in rows:
            vec = json.loads(row["embedding"])
            if len(vec) != dim:
                continue
            self._conn.execute(
                f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
                (row["rowid"], vec_mod.serialize_f32(vec)),
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
                 is_deleted      = 0,
                 updated_at      = datetime('now')""",
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

        if self._vec_enabled and embedding:
            dim = len(embedding)
            ready = vec_mod.ensure_table(self._conn, item["dataset_key"], dim)
            if ready:
                rowid = self._rowid_for_item(item["id"])
                if rowid is not None:
                    vec_mod.upsert_vector(self._conn, item["dataset_key"], rowid, embedding)
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
            "UPDATE memory_items SET is_deleted = 1, updated_at = datetime('now') WHERE id = ? AND is_deleted = 0",
            (item_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_memory_item(self, item_id: str) -> bool:
        """Hard delete — permanently removes the row, its vec0 entry, and any
        relationships that reference this item as source or target."""
        row = self._conn.execute(
            "SELECT rowid, dataset_key FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        if not row:
            return False

        # Cascade: remove any relationship edges that reference this item.
        self._conn.execute(
            "DELETE FROM relationships WHERE source_key = ? OR target_key = ?",
            (item_id, item_id),
        )

        cur = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()

        if cur.rowcount > 0 and self._vec_enabled:
            vec_mod.delete_vector(self._conn, row["dataset_key"], row["rowid"])
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
            return self._search_vector_only(
                dataset_key, query_vector, top_k, metadata_filters  # type: ignore[arg-type]
            )

        if query_vector is None and keyword_query is not None:
            return self._search_keyword_only(dataset_key, keyword_query, top_k, metadata_filters)

        if needs_vector and keyword_query is not None:
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
            if all(self._matches_filter(it, k, v) for k, v in metadata_filters.items())
        ]

    @staticmethod
    def _filter_value(item: dict[str, Any], key: str) -> Any:
        if key in ("created_at", "updated_at"):
            return item.get(key)
        return item.get("metadata", {}).get(key)

    @staticmethod
    def _compare_value(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return value
        raw = value.strip()
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw

    @classmethod
    def _ordered_compare(cls, left: Any, right: Any, op: str) -> bool:
        if left is None or right is None:
            return False
        left_value = cls._compare_value(left)
        right_value = cls._compare_value(right)
        if isinstance(left_value, datetime) and isinstance(right_value, datetime):
            if (left_value.tzinfo is None) != (right_value.tzinfo is None):
                left_value = left_value.replace(tzinfo=None)
                right_value = right_value.replace(tzinfo=None)
        try:
            if op == "$gte":
                return left_value >= right_value
            if op == "$lte":
                return left_value <= right_value
        except TypeError:
            return False
        return False

    @classmethod
    def _matches_filter(cls, item: dict[str, Any], key: str, criterion: Any) -> bool:
        value = cls._filter_value(item, key)
        if isinstance(criterion, dict):
            if not criterion:
                return False
            for op, expected in criterion.items():
                if op == "$gte":
                    if not cls._ordered_compare(value, expected, op):
                        return False
                elif op == "$lte":
                    if not cls._ordered_compare(value, expected, op):
                        return False
                elif op == "$between":
                    if (
                        not isinstance(expected, list)
                        or len(expected) != 2
                        or not cls._ordered_compare(value, expected[0], "$gte")
                        or not cls._ordered_compare(value, expected[1], "$lte")
                    ):
                        return False
                elif op == "$in":
                    if not isinstance(expected, list) or value not in expected:
                        return False
                else:
                    return False
            return True
        return value == criterion

    def _search_vector_only(
        self,
        dataset_key: str,
        query_vector: list[float],
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self._vec_enabled and vec_mod.table_exists(self._conn, dataset_key):
            return self._vec_knn_search(
                dataset_key, query_vector, top_k, metadata_filters,
                keyword_query=None, vector_weight=1.0
            )
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
        scored = bm25_score(items, keyword_query)
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
        if self._vec_enabled and vec_mod.table_exists(self._conn, dataset_key):
            return self._vec_knn_search(
                dataset_key, query_vector, top_k, metadata_filters,
                keyword_query=keyword_query, vector_weight=vector_weight
            )
        items = self._load_items_for_scan(dataset_key, require_embedding=False)
        items = self._apply_metadata_filter(items, metadata_filters)

        items_with_kw = bm25_score(items, keyword_query)

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
        """ANN search via vec0 with optional BM25 blending and metadata filtering."""
        tbl = vec_mod.vec_table_name(dataset_key)
        total = self.count_memory_items(dataset_key)
        candidate_k = min(total, max(top_k * 20, top_k))

        knn_rows = self._conn.execute(
            f"SELECT rowid, distance FROM {tbl} "
            f"WHERE embedding MATCH ? AND k = ?",
            (vec_mod.serialize_f32(query_vector), candidate_k),
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
            items = bm25_score(items, keyword_query)

        scored = []
        for item in items:
            item.pop("_embedding_raw", None)
            rowid = item.pop("_rowid", None)
            ann_dist = rowid_to_dist.get(rowid, 1.0)
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
            "UPDATE memory_items SET embedding = ?, embedding_model = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(embedding), model_id, item_id),
        )
        self._conn.commit()

        if self._vec_enabled:
            row = self._conn.execute(
                "SELECT rowid, dataset_key FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row:
                dataset_key = row["dataset_key"]
                rowid = row["rowid"]
                dim = len(embedding)
                ready = vec_mod.ensure_table(self._conn, dataset_key, dim)
                if ready:
                    vec_mod.upsert_vector(self._conn, dataset_key, rowid, embedding)
                    self._conn.commit()

    def _rowid_for_item(self, item_id: str) -> int | None:
        row = self._conn.execute(
            "SELECT rowid FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        raw_emb = d.pop("embedding", None)
        d["_embedding_raw"] = json.loads(raw_emb) if raw_emb else None
        d["is_deleted"] = bool(d.get("is_deleted", 0))
        d.pop("rowid", None)
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
