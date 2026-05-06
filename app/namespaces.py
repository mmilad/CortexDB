"""Namespace and subspace-scoped SQLite stores.

A namespace maps to one SQLite database file, and each namespace can also own
one level of subspace database files. This gives strong local isolation while
keeping the API surface identical under ``/{namespace}`` and
``/{namespace}/{subspace}``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import HTTPException, Request

from app.store import SqliteStore, get_store

_NAMESPACE_ENV_VAR = "CORTEXDB_NAMESPACE_DIR"
_DB_PATH_ENV_VAR = "CORTEXDB_DB_PATH"
_DEFAULT_NAMESPACE_DIR = ".cortexdb/namespaces"
_VALID_NAMESPACE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_RESERVED_PATH_PARTS = {
    "capabilities",
    "context",
    "datasets",
    "docs",
    "graph",
    "health",
    "mcp",
    "namespaces",
    "new_subspace",
    "openapi.json",
    "relationships",
    "subspaces",
    "tools",
}

_stores: dict[str, SqliteStore] = {}


def validate_namespace(name: str) -> str:
    if not _VALID_NAMESPACE.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail=(
                "namespace must be 1-64 chars and contain only letters, "
                "numbers, underscores, or hyphens; it must start with a letter or number"
            ),
        )
    if name.lower() in _RESERVED_PATH_PARTS:
        raise HTTPException(
            status_code=422,
            detail=f"namespace or subspace name is reserved: {name}",
        )
    return name


def namespace_dir() -> Path:
    configured = os.environ.get(_NAMESPACE_ENV_VAR)
    if configured:
        return Path(configured)

    db_path = Path(os.environ.get(_DB_PATH_ENV_VAR, "cortexdb.sqlite"))
    parent = db_path.parent if str(db_path.parent) not in ("", ".") else Path.cwd()
    return parent / _DEFAULT_NAMESPACE_DIR


def namespace_db_path(namespace: str) -> Path:
    namespace = validate_namespace(namespace)
    base = namespace_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{namespace}.sqlite"


def subspace_db_path(namespace: str, subspace: str) -> Path:
    namespace = validate_namespace(namespace)
    subspace = validate_namespace(subspace)
    base = namespace_dir() / namespace
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{subspace}.sqlite"


def create_namespace(namespace: str) -> Path:
    path = namespace_db_path(namespace)
    store = get_namespace_store(namespace)
    store.close()
    _stores.pop(_store_key(namespace), None)
    return path


def create_subspace(namespace: str, subspace: str) -> Path:
    path = subspace_db_path(namespace, subspace)
    store = get_namespace_store(namespace, subspace)
    store.close()
    _stores.pop(_store_key(namespace, subspace), None)
    return path


def list_namespaces() -> list[str]:
    base = namespace_dir()
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.sqlite") if _VALID_NAMESPACE.fullmatch(p.stem))


def list_subspaces(namespace: str) -> list[str]:
    namespace = validate_namespace(namespace)
    base = namespace_dir() / namespace
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.sqlite") if _VALID_NAMESPACE.fullmatch(p.stem))


def _store_key(namespace: str, subspace: str | None = None) -> str:
    if subspace is None:
        return namespace
    return f"{namespace}/{subspace}"


def get_namespace_store(namespace: str, subspace: str | None = None) -> SqliteStore:
    namespace = validate_namespace(namespace)
    if subspace is not None:
        subspace = validate_namespace(subspace)

    key = _store_key(namespace, subspace)
    store = _stores.get(key)
    if store is None:
        path = subspace_db_path(namespace, subspace) if subspace else namespace_db_path(namespace)
        store = SqliteStore(str(path))
        _stores[key] = store
    return store


def get_store_for_request(request: Request) -> SqliteStore:
    namespace = request.path_params.get("namespace")
    if namespace:
        subspace = request.path_params.get("subspace")
        return get_namespace_store(str(namespace), str(subspace) if subspace else None)
    return get_store()


def close_namespace_stores() -> None:
    for store in _stores.values():
        store.close()
    _stores.clear()
