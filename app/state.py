"""Registry state backed by SQLite via app.store.

Public API is unchanged (RegistryState + get_registry) so existing API handlers
require no edits. The in-memory fallback is gone; all data is now persisted.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends

from app.store import SqliteStore, get_store


class RegistryState:
    """Thin proxy that surfaces SqliteStore as dict-like .datasets / .tools attributes.

    This preserves backward-compatibility with existing API handlers that do:
        reg.datasets[key] = record.model_dump()
        reg.tools.get(key)
    etc., while persisting all data to SQLite underneath.
    """

    def __init__(self, store: SqliteStore) -> None:
        self._store = store
        self.datasets = _DatasetProxy(store)
        self.tools = _ToolProxy(store)


class _DatasetProxy:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        self._store.upsert_dataset(key, value)

    def __getitem__(self, key: str) -> dict[str, Any]:
        val = self._store.get_dataset(key)
        if val is None:
            raise KeyError(key)
        return val

    def get(self, key: str, default: Any = None) -> dict[str, Any] | None:
        val = self._store.get_dataset(key)
        return val if val is not None else default

    def keys(self) -> list[str]:
        return list(self._store.list_datasets().keys())

    def values(self) -> list[dict[str, Any]]:
        return list(self._store.list_datasets().values())

    def items(self):
        return self._store.list_datasets().items()

    def __contains__(self, key: str) -> bool:
        return self._store.get_dataset(key) is not None


class _ToolProxy:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        self._store.upsert_tool(key, value)

    def __getitem__(self, key: str) -> dict[str, Any]:
        val = self._store.get_tool(key)
        if val is None:
            raise KeyError(key)
        return val

    def get(self, key: str, default: Any = None) -> dict[str, Any] | None:
        val = self._store.get_tool(key)
        return val if val is not None else default

    def keys(self) -> list[str]:
        return list(self._store.list_tools().keys())

    def values(self) -> list[dict[str, Any]]:
        return list(self._store.list_tools().values())

    def items(self):
        return self._store.list_tools().items()

    def __contains__(self, key: str) -> bool:
        return self._store.get_tool(key) is not None


def get_registry(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> RegistryState:
    return RegistryState(store)
