"""sqlite-vec virtual-table helpers.

All functions operate on an existing sqlite3.Connection and are called only
when the sqlite-vec extension has been successfully loaded.
"""

from __future__ import annotations

import importlib
import logging
import re
import sqlite3
import struct

logger = logging.getLogger("cortexdb.store.vec")


def try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
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


def vec_table_name(dataset_key: str) -> str:
    """Safe vec0 table name derived from a dataset key."""
    safe = re.sub(r"[^a-z0-9]", "_", dataset_key.lower())
    return f"vec_items_{safe}"


def serialize_f32(vec: list[float]) -> bytes:
    """Convert a float list to sqlite-vec float32 binary format."""
    return struct.pack(f"{len(vec)}f", *vec)


def table_exists(conn: sqlite3.Connection, dataset_key: str) -> bool:
    tbl = vec_table_name(dataset_key)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
    ).fetchone()
    return row is not None


def create_table(conn: sqlite3.Connection, dataset_key: str, dim: int) -> None:
    tbl = vec_table_name(dataset_key)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {tbl} "
        f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    conn.execute(
        "UPDATE datasets SET vec_dim = ? WHERE dataset_key = ?", (dim, dataset_key)
    )
    conn.commit()


def drop_table(conn: sqlite3.Connection, dataset_key: str) -> None:
    tbl = vec_table_name(dataset_key)
    conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.execute(
        "UPDATE datasets SET vec_dim = NULL WHERE dataset_key = ?", (dataset_key,)
    )
    conn.commit()


def get_dim(conn: sqlite3.Connection, dataset_key: str) -> int | None:
    row = conn.execute(
        "SELECT vec_dim FROM datasets WHERE dataset_key = ?", (dataset_key,)
    ).fetchone()
    return row["vec_dim"] if row else None


def ensure_table(conn: sqlite3.Connection, dataset_key: str, dim: int) -> bool:
    """Ensure a vec0 table exists at the requested dimension.

    Returns True if ready, False if the stored dimension differs (caller must
    call rebuild to reconcile).
    """
    stored_dim = get_dim(conn, dataset_key)
    if stored_dim is None:
        create_table(conn, dataset_key, dim)
        return True
    if stored_dim != dim:
        return False
    if not table_exists(conn, dataset_key):
        create_table(conn, dataset_key, dim)
    return True


def upsert_vector(
    conn: sqlite3.Connection, dataset_key: str, rowid: int, embedding: list[float]
) -> None:
    """Insert or replace a vector (vec0 requires DELETE + INSERT)."""
    tbl = vec_table_name(dataset_key)
    conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (rowid,))
    conn.execute(
        f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
        (rowid, serialize_f32(embedding)),
    )


def delete_vector(conn: sqlite3.Connection, dataset_key: str, rowid: int) -> None:
    if table_exists(conn, dataset_key):
        tbl = vec_table_name(dataset_key)
        conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (rowid,))
