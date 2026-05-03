"""SQLite-backed persistent store for CortexDB registry data.

Uses Python stdlib sqlite3 only — no extra dependencies.
WAL mode is enabled for safe concurrent reads under light write load.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_DB_PATH_ENV_VAR = "CORTEXDB_DB_PATH"
_DEFAULT_DB_PATH = "cortexdb.sqlite"

_DDL = """\
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_key TEXT PRIMARY KEY,
    data        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tools (
    tool_key TEXT PRIMARY KEY,
    data     TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships (source_type, source_key);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships (target_type, target_key);
"""


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

    def get_dataset(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM datasets WHERE dataset_key = ?", (key,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT dataset_key, data FROM datasets").fetchall()
        return {r["dataset_key"]: json.loads(r["data"]) for r in rows}

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
        """Return all edges for graph traversal."""
        rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        return [self._row_to_rel(r) for r in rows]

    @staticmethod
    def _row_to_rel(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["join_fields"] = json.loads(d["join_fields"])
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
