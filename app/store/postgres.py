"""PostgreSQL + pgvector storage backend for CortexDB.

The SQLite backend remains the default for local tests.  This adapter keeps
the same small store surface used by the HTTP API, while moving JSON metadata
into jsonb and embeddings into pgvector.  Namespace stores use a dedicated
Postgres schema, which preserves CortexDB's existing isolation model without
creating a database per namespace.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from app.store.search import bm25_score, cosine_similarity

try:  # Keep SQLite-only installs usable without the optional extra.
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover - exercised only when Postgres is requested
    psycopg = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    register_vector = None  # type: ignore[assignment]


_SCHEMA_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def _json(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip("[]")
        return [float(part) for part in raw.split(",") if part.strip()]
    return [float(part) for part in value]


class PostgresStore:
    """Synchronous Postgres store with the CortexDB store contract."""

    def __init__(self, database_url: str | None = None, *, schema: str = "cortexdb") -> None:
        if psycopg is None or sql is None or dict_row is None or register_vector is None:
            raise RuntimeError(
                "PostgreSQL support requires the optional dependencies. "
                "Install CortexDB with: pip install -e '.[postgres]'"
            )
        if not _SCHEMA_RE.fullmatch(schema):
            raise ValueError(f"invalid PostgreSQL schema name: {schema!r}")
        self.schema = schema
        self._conn = psycopg.connect(database_url or os.environ["CORTEXDB_DATABASE_URL"], row_factory=dict_row)
        self._conn.autocommit = False
        register_vector(self._conn)
        self._ensure_schema()

    def _table(self, name: str) -> sql.Composed:
        return sql.SQL(".").join((sql.Identifier(self.schema), sql.Identifier(name)))

    def _ensure_schema(self) -> None:
        schema = sql.Identifier(self.schema)
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}" ).format(schema))
            cur.execute(sql.SQL("SET search_path TO {}, public").format(schema))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {datasets} (
                        dataset_key TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        embed_raw TEXT,
                        embedding vector,
                        embedding_model TEXT,
                        embedded_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        vec_dim INTEGER
                    );
                    CREATE TABLE IF NOT EXISTS {tools} (
                        tool_key TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {rule_packs} (
                        key TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'active',
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (namespace, key)
                    );
                    CREATE TABLE IF NOT EXISTS {relationships} (
                        id TEXT PRIMARY KEY,
                        source_type TEXT NOT NULL,
                        source_key TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        target_key TEXT NOT NULL,
                        edge_type TEXT NOT NULL,
                        join_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
                        description TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {sessions} (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL DEFAULT 'chat',
                        scope_mode TEXT NOT NULL DEFAULT 'namespace',
                        namespace TEXT,
                        dataset_policy TEXT NOT NULL DEFAULT 'create_if_needed',
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {raw_texts} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'unknown',
                        relations JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        score DOUBLE PRECISION,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector,
                        embedding_model TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {summaries} (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES {sessions}(id) ON DELETE CASCADE,
                        summary TEXT NOT NULL,
                        message_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                        token_estimate INTEGER NOT NULL DEFAULT 0,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {messages} (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES {sessions}(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        raw_text_id TEXT,
                        token_estimate INTEGER NOT NULL DEFAULT 0,
                        autocontext_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        summary_id TEXT REFERENCES {summaries}(id) ON DELETE SET NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS {items} (
                        id TEXT PRIMARY KEY,
                        dataset_key TEXT NOT NULL REFERENCES {datasets}(dataset_key) ON DELETE CASCADE,
                        raw_text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector,
                        embedding_model TEXT,
                        is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS {items_dataset_idx} ON {items} (dataset_key, is_deleted);
                    CREATE INDEX IF NOT EXISTS {messages_session_idx} ON {messages} (session_id, created_at);
                    CREATE INDEX IF NOT EXISTS {summaries_session_idx} ON {summaries} (session_id, created_at);
                    CREATE INDEX IF NOT EXISTS {relationships_source_idx} ON {relationships} (source_type, source_key);
                    CREATE INDEX IF NOT EXISTS {relationships_target_idx} ON {relationships} (target_type, target_key);
                    """
                ).format(
                    datasets=self._table("datasets"),
                    tools=self._table("tools"),
                    rule_packs=self._table("ingest_rule_packs"),
                    relationships=self._table("relationships"),
                    sessions=self._table("sessions"),
                    raw_texts=self._table("raw_texts"),
                    summaries=self._table("session_summaries"),
                    messages=self._table("session_messages"),
                    items=self._table("memory_items"),
                    items_dataset_idx=sql.Identifier(f"{self.schema}_items_dataset_idx"),
                    messages_session_idx=sql.Identifier(f"{self.schema}_messages_session_idx"),
                    summaries_session_idx=sql.Identifier(f"{self.schema}_summaries_session_idx"),
                    relationships_source_idx=sql.Identifier(f"{self.schema}_relationships_source_idx"),
                    relationships_target_idx=sql.Identifier(f"{self.schema}_relationships_target_idx"),
                )
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def vec_enabled(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def upsert_dataset(self, key: str, data: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """INSERT INTO {table} (dataset_key, data)
                       VALUES (%s, %s)
                       ON CONFLICT (dataset_key) DO UPDATE SET
                         data = EXCLUDED.data, updated_at = CURRENT_TIMESTAMP"""
                ).format(table=self._table("datasets")),
                (key, data),
            )
        self._conn.commit()

    def set_dataset_embedding(self, key: str, raw_text: str, embedding: list[float], model_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """UPDATE {table}
                       SET embed_raw = %s, embedding = %s, embedding_model = %s,
                           embedded_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                           vec_dim = %s WHERE dataset_key = %s"""
                ).format(table=self._table("datasets")),
                (raw_text, embedding, model_id, len(embedding), key),
            )
        self._conn.commit()

    def get_dataset(self, key: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT data, created_at, updated_at FROM {table} WHERE dataset_key = %s").format(table=self._table("datasets")), (key,))
            row = cur.fetchone()
        if row is None:
            return None
        data = dict(_json(row["data"]))
        data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
        data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
        return data

    def get_dataset_embedding(self, key: str) -> tuple[list[float] | None, str | None]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT embedding, embedding_model FROM {table} WHERE dataset_key = %s").format(table=self._table("datasets")), (key,))
            row = cur.fetchone()
        return (_vector(row["embedding"]), row["embedding_model"]) if row and row["embedding"] is not None else (None, None)

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT dataset_key, data, created_at, updated_at FROM {table} ORDER BY dataset_key").format(table=self._table("datasets")))
            rows = cur.fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = dict(_json(row["data"]))
            data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
            data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
            result[row["dataset_key"]] = data
        return result

    def list_datasets_with_embeddings(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT dataset_key, data, embedding, embedding_model, created_at, updated_at FROM {table} WHERE embedding IS NOT NULL").format(table=self._table("datasets")))
            rows = cur.fetchall()
        result = []
        for row in rows:
            data = dict(_json(row["data"]))
            data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
            data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
            result.append({"dataset_key": row["dataset_key"], "data": data, "embedding": _vector(row["embedding"]), "embedding_model": row["embedding_model"]})
        return result

    def delete_dataset(self, key: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {table} WHERE dataset_key = %s").format(table=self._table("datasets")), (key,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def upsert_tool(self, key: str, data: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (tool_key, data) VALUES (%s, %s)
                ON CONFLICT (tool_key) DO UPDATE SET data = EXCLUDED.data, updated_at = CURRENT_TIMESTAMP""").format(table=self._table("tools")), (key, data))
        self._conn.commit()

    def get_tool(self, key: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT data, created_at, updated_at FROM {table} WHERE tool_key = %s").format(table=self._table("tools")), (key,))
            row = cur.fetchone()
        if row is None:
            return None
        data = dict(_json(row["data"]))
        data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
        data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
        return data

    def list_tools(self) -> dict[str, dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT tool_key, data, created_at, updated_at FROM {table} ORDER BY tool_key").format(table=self._table("tools")))
            rows = cur.fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = dict(_json(row["data"]))
            data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
            data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
            result[row["tool_key"]] = data
        return result

    def delete_tool(self, key: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {table} WHERE tool_key = %s").format(table=self._table("tools")), (key,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    # ------------------------------------------------------------------
    # Rule packs and relationships
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace_key(namespace: str | None) -> str:
        return namespace or ""

    def upsert_ingest_rule_pack(self, key: str, data: dict[str, Any], namespace: str | None = None) -> None:
        ns = self._namespace_key(namespace)
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (key, namespace, status, data) VALUES (%s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE SET status = EXCLUDED.status, data = EXCLUDED.data, updated_at = CURRENT_TIMESTAMP""").format(table=self._table("ingest_rule_packs")), (key, ns, data.get("status", "active"), data))
        self._conn.commit()

    def get_ingest_rule_pack(self, key: str, namespace: str | None = None) -> dict[str, Any] | None:
        ns = self._namespace_key(namespace)
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT key, namespace, data, created_at, updated_at FROM {table} WHERE namespace = %s AND key = %s").format(table=self._table("ingest_rule_packs")), (ns, key))
            row = cur.fetchone()
        return self._row_to_rule_pack(row) if row else None

    def list_ingest_rule_packs(self, namespace: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        ns = self._namespace_key(namespace)
        where = "namespace = %s" + (" AND status = 'active'" if active_only else "")
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT key, namespace, data, created_at, updated_at FROM {table} WHERE " + where + " ORDER BY key").format(table=self._table("ingest_rule_packs")), (ns,))
            rows = cur.fetchall()
        return [self._row_to_rule_pack(row) for row in rows]

    def delete_ingest_rule_pack(self, key: str, namespace: str | None = None) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {table} WHERE namespace = %s AND key = %s").format(table=self._table("ingest_rule_packs")), (self._namespace_key(namespace), key))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    @staticmethod
    def _row_to_rule_pack(row: dict[str, Any]) -> dict[str, Any]:
        data = dict(_json(row["data"]))
        data.setdefault("key", row["key"])
        data["namespace"] = data.get("namespace") or row["namespace"] or None
        data.setdefault("created_at", row["created_at"].isoformat() if row["created_at"] else None)
        data.setdefault("updated_at", row["updated_at"].isoformat() if row["updated_at"] else None)
        return data

    def upsert_relationship(self, rel: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, source_type, source_key, target_type, target_key, edge_type, join_fields, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET source_type = EXCLUDED.source_type, source_key = EXCLUDED.source_key,
                  target_type = EXCLUDED.target_type, target_key = EXCLUDED.target_key, edge_type = EXCLUDED.edge_type,
                  join_fields = EXCLUDED.join_fields, description = EXCLUDED.description""").format(table=self._table("relationships")), (rel["id"], rel["source_type"], rel["source_key"], rel["target_type"], rel["target_key"], rel["edge_type"], rel.get("join_fields", []), rel.get("description", "")))
        self._conn.commit()

    def get_relationship(self, rel_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE id = %s").format(table=self._table("relationships")), (rel_id,))
            return cur.fetchone()

    def list_relationships(self, source_key: str | None = None, target_key: str | None = None) -> list[dict[str, Any]]:
        if source_key or target_key:
            key = source_key or target_key
            query = sql.SQL("SELECT * FROM {table} WHERE source_key = %s OR target_key = %s").format(table=self._table("relationships"))
            params = (key, key)
        else:
            query = sql.SQL("SELECT * FROM {table} ORDER BY id").format(table=self._table("relationships"))
            params = ()
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        for row in rows:
            row["join_fields"] = _json(row["join_fields"])
        return rows

    def delete_relationship(self, rel_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {table} WHERE id = %s").format(table=self._table("relationships")), (rel_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def adjacency(self) -> list[dict[str, Any]]:
        return self.list_relationships()

    # ------------------------------------------------------------------
    # Sessions and source audit
    # ------------------------------------------------------------------

    def upsert_session(self, session: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, type, scope_mode, namespace, dataset_policy, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET type = EXCLUDED.type, scope_mode = EXCLUDED.scope_mode,
                  namespace = EXCLUDED.namespace, dataset_policy = EXCLUDED.dataset_policy,
                  metadata = EXCLUDED.metadata, updated_at = CURRENT_TIMESTAMP""").format(table=self._table("sessions")), (session["id"], session.get("type", "chat"), session.get("scope_mode", "namespace"), session.get("namespace"), session.get("dataset_policy", "create_if_needed"), session.get("metadata", {})))
        self._conn.commit()

    def ensure_session(self, session_id: str = "main", *, type: str = "chat", scope_mode: str = "namespace", namespace: str | None = None, dataset_policy: str = "create_if_needed", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        existing = self.get_session(session_id)
        if existing:
            return existing
        self.upsert_session({"id": session_id, "type": type, "scope_mode": scope_mode, "namespace": namespace, "dataset_policy": dataset_policy, "metadata": metadata or {}})
        created = self.get_session(session_id)
        if created is None:
            raise RuntimeError(f"could not create session '{session_id}'")
        return created

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE id = %s").format(table=self._table("sessions")), (session_id,))
            row = cur.fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} ORDER BY updated_at DESC").format(table=self._table("sessions")))
            rows = cur.fetchall()
        return [self._row_to_session(row) for row in rows]

    def update_session(self, session_id: str, *, type: str | None = None, scope_mode: str | None = None, namespace: str | None = None, dataset_policy: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        existing = self.get_session(session_id)
        if existing is None:
            return None
        self.upsert_session({"id": session_id, "type": type or existing["type"], "scope_mode": scope_mode or existing["scope_mode"], "namespace": namespace if namespace is not None else existing["namespace"], "dataset_policy": dataset_policy or existing["dataset_policy"], "metadata": metadata if metadata is not None else existing["metadata"]})
        return self.get_session(session_id)

    def rename_session(self, session_id: str, new_session_id: str) -> dict[str, Any] | None:
        if session_id == new_session_id:
            return self.get_session(session_id)
        if self.get_session(session_id) is None or self.get_session(new_session_id) is not None:
            return None
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {table} SET id = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s").format(table=self._table("sessions")), (new_session_id, session_id))
            cur.execute(sql.SQL("UPDATE {table} SET session_id = %s WHERE session_id = %s").format(table=self._table("session_messages")), (new_session_id, session_id))
            cur.execute(sql.SQL("UPDATE {table} SET session_id = %s WHERE session_id = %s").format(table=self._table("session_summaries")), (new_session_id, session_id))
        self._conn.commit()
        return self.get_session(new_session_id)

    def delete_session(self, session_id: str, *, delete_related_chunks: bool = False) -> bool:
        if self.get_session(session_id) is None:
            return False
        with self._conn.cursor() as cur:
            if delete_related_chunks:
                cur.execute(sql.SQL("DELETE FROM {table} WHERE id IN (SELECT raw_text_id FROM {messages} WHERE session_id = %s AND raw_text_id IS NOT NULL)").format(table=self._table("raw_texts"), messages=self._table("session_messages")), (session_id,))
            cur.execute(sql.SQL("DELETE FROM {table} WHERE id = %s").format(table=self._table("sessions")), (session_id,))
        self._conn.commit()
        return True

    def insert_raw_text(self, raw: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, text, source, relations, score, metadata, embedding, embedding_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""").format(table=self._table("raw_texts")), (raw["id"], raw["text"], raw.get("source", "unknown"), raw.get("relations", {}), raw.get("score"), raw.get("metadata", {}), raw.get("embedding"), raw.get("embedding_model")))
        self._conn.commit()

    def get_raw_text(self, raw_text_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE id = %s").format(table=self._table("raw_texts")), (raw_text_id,))
            row = cur.fetchone()
        return self._row_to_raw_text(row) if row else None

    def insert_session_message(self, message: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, session_id, role, content, raw_text_id, token_estimate, autocontext_enabled, summary_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""").format(table=self._table("session_messages")), (message["id"], message["session_id"], message.get("role", "user"), message["content"], message.get("raw_text_id"), message.get("token_estimate", 0), message.get("autocontext_enabled", True), message.get("summary_id"), message.get("metadata", {})))
        self._conn.commit()

    def list_session_messages(self, session_id: str, *, limit: int = 100, offset: int = 0, autocontext_only: bool = False) -> list[dict[str, Any]]:
        extra = " AND autocontext_enabled = TRUE" if autocontext_only else ""
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE session_id = %s" + extra + " ORDER BY created_at LIMIT %s OFFSET %s").format(table=self._table("session_messages")), (session_id, limit, offset))
            rows = cur.fetchall()
        return [self._row_to_session_message(row) for row in rows]

    def insert_session_summary(self, summary: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, session_id, summary, message_ids, token_estimate, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)""").format(table=self._table("session_summaries")), (summary["id"], summary["session_id"], summary["summary"], summary.get("message_ids", []), summary.get("token_estimate", 0), summary.get("metadata", {})))
        self._conn.commit()

    def list_session_summaries(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE session_id = %s ORDER BY created_at").format(table=self._table("session_summaries")), (session_id,))
            rows = cur.fetchall()
        return [self._row_to_session_summary(row) for row in rows]

    def disable_session_messages_for_autocontext(self, session_id: str, message_ids: list[str], summary_id: str) -> None:
        if not message_ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {table} SET autocontext_enabled = FALSE, summary_id = %s WHERE session_id = %s AND id = ANY(%s)").format(table=self._table("session_messages")), (summary_id, session_id, message_ids))
        self._conn.commit()

    @staticmethod
    def _row_to_session(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = _json(result.get("metadata"))
        for key in ("created_at", "updated_at"):
            if result.get(key):
                result[key] = result[key].isoformat()
        return result

    @staticmethod
    def _row_to_raw_text(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["relations"] = _json(result.get("relations"))
        result["metadata"] = _json(result.get("metadata"))
        result["embedding"] = _vector(result.get("embedding"))
        if result.get("created_at"):
            result["created_at"] = result["created_at"].isoformat()
        return result

    @staticmethod
    def _row_to_session_message(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = _json(result.get("metadata"))
        if result.get("created_at"):
            result["created_at"] = result["created_at"].isoformat()
        return result

    @staticmethod
    def _row_to_session_summary(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["message_ids"] = _json(result.get("message_ids"))
        result["metadata"] = _json(result.get("metadata"))
        if result.get("created_at"):
            result["created_at"] = result["created_at"].isoformat()
        return result

    # ------------------------------------------------------------------
    # Memory and retrieval
    # ------------------------------------------------------------------

    def rebuild_vec_index(self, dataset_key: str) -> int:
        # pgvector indexes are intentionally created only after measuring the
        # real deployment. The vector column itself is the source of truth.
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) AS n FROM {table} WHERE dataset_key = %s AND embedding IS NOT NULL AND NOT is_deleted").format(table=self._table("memory_items")), (dataset_key,))
            row = cur.fetchone()
        return int(row["n"] if row else 0)

    def insert_memory_item(self, item: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""INSERT INTO {table} (id, dataset_key, raw_text, metadata, embedding, embedding_model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET dataset_key = EXCLUDED.dataset_key, raw_text = EXCLUDED.raw_text,
                  metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model,
                  updated_at = CURRENT_TIMESTAMP""").format(table=self._table("memory_items")), (item["id"], item["dataset_key"], item["raw_text"], item.get("metadata", {}), item.get("embedding"), item.get("embedding_model")))
        self._conn.commit()

    def list_memory_items(self, dataset_key: str, limit: int = 100, offset: int = 0, include_deleted: bool = False) -> list[dict[str, Any]]:
        extra = "" if include_deleted else " AND NOT is_deleted"
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE dataset_key = %s" + extra + " ORDER BY created_at DESC LIMIT %s OFFSET %s").format(table=self._table("memory_items")), (dataset_key, limit, offset))
            rows = cur.fetchall()
        return [self._row_to_item(row) for row in rows]

    def list_all_memory_items(self, dataset_key: str) -> list[dict[str, Any]]:
        return self.list_memory_items(dataset_key, limit=2_147_483_647)

    def get_memory_item(self, item_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT * FROM {table} WHERE id = %s").format(table=self._table("memory_items")), (item_id,))
            row = cur.fetchone()
        return self._row_to_item(row) if row else None

    def soft_delete_memory_item(self, item_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {table} SET is_deleted = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = %s AND NOT is_deleted").format(table=self._table("memory_items")), (item_id,))
            changed = cur.rowcount > 0
        self._conn.commit()
        return changed

    def delete_memory_item(self, item_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {rel} WHERE source_key = %s OR target_key = %s").format(rel=self._table("relationships")), (item_id, item_id))
            cur.execute(sql.SQL("DELETE FROM {items} WHERE id = %s").format(items=self._table("memory_items")), (item_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def count_memory_items(self, dataset_key: str, include_deleted: bool = False) -> int:
        extra = "" if include_deleted else " AND NOT is_deleted"
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) AS n FROM {table} WHERE dataset_key = %s" + extra).format(table=self._table("memory_items")), (dataset_key,))
            row = cur.fetchone()
        return int(row["n"] if row else 0)

    @classmethod
    def _filter_value(cls, item: dict[str, Any], key: str) -> Any:
        if key in item:
            return item[key]
        if key in item.get("metadata", {}):
            return item["metadata"][key]
        return None

    @classmethod
    def _matches_filter(cls, item: dict[str, Any], key: str, criterion: Any) -> bool:
        value = cls._filter_value(item, key)
        if isinstance(criterion, dict):
            for op, expected in criterion.items():
                if op == "$in" and value not in expected:
                    return False
                if op == "$eq" and value != expected:
                    return False
                if op == "$gte" and (value is None or value < expected):
                    return False
                if op == "$lte" and (value is None or value > expected):
                    return False
                if op == "$gt" and (value is None or value <= expected):
                    return False
                if op == "$lt" and (value is None or value >= expected):
                    return False
                if op == "$between" and (not isinstance(expected, list) or len(expected) != 2 or value is None or value < expected[0] or value > expected[1]):
                    return False
            return True
        return value == criterion

    def search_memory_items(self, dataset_key: str, query_vector: list[float] | None = None, top_k: int = 10, metadata_filters: dict[str, Any] | None = None, keyword_query: str | None = None, vector_weight: float = 1.0) -> list[dict[str, Any]]:
        vector_select = ""
        params: list[Any] = []
        if query_vector is not None:
            # pgvector computes the distance in PostgreSQL. The Python cosine
            # fallback below only handles rows without a usable score.
            vector_select = ", (1 - (embedding <=> %s)) AS db_vector_score"
            params.append(query_vector)
        params.append(dataset_key)
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT *" + vector_select + " FROM {table} WHERE dataset_key = %s AND NOT is_deleted").format(table=self._table("memory_items")), tuple(params))
            rows = cur.fetchall()
        items = [self._row_to_item(row) for row in rows]
        if metadata_filters:
            items = [item for item in items if all(self._matches_filter(item, key, criterion) for key, criterion in metadata_filters.items())]
        for item in items:
            raw_vector = item.get("_embedding_raw")
            db_score = item.pop("db_vector_score", None)
            item["vector_score"] = float(db_score) if db_score is not None else (cosine_similarity(query_vector, raw_vector) if query_vector and raw_vector else 0.0)
        if keyword_query:
            items = bm25_score(items, keyword_query)
            for item in items:
                item["keyword_score"] = item.get("score", 0.0)
        else:
            for item in items:
                item["keyword_score"] = 0.0
        for item in items:
            item["score"] = vector_weight * item.get("vector_score", 0.0) + (1.0 - vector_weight) * item.get("keyword_score", 0.0)
        items.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return items[:top_k]

    def update_memory_item_embedding(self, item_id: str, embedding: list[float], model_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {table} SET embedding = %s, embedding_model = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s").format(table=self._table("memory_items")), (embedding, model_id, item_id))
        self._conn.commit()

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = _json(result.get("metadata"))
        result["_embedding_raw"] = _vector(result.pop("embedding", None))
        result["is_deleted"] = bool(result.get("is_deleted"))
        for key in ("created_at", "updated_at"):
            if result.get(key):
                result[key] = result[key].isoformat()
        return result


def postgres_schema_for_namespace(namespace: str | None, subspace: str | None = None) -> str:
    parts = [part for part in (namespace, subspace) if part]
    suffix = "_".join(re.sub(r"[^a-zA-Z0-9_]", "_", part) for part in parts)
    return "cortexdb" if not suffix else f"cortexdb_{suffix}"[:63]
