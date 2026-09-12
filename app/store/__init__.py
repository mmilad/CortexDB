"""CortexDB storage package.

Public surface is identical to the old monolithic app/store.py so all
import sites (app/api/*, app/mcp/*, tests/*) continue to work unchanged.

To reset the singleton in tests, import from app.store.main directly:
    import app.store.main as store_mod
    store_mod._store = None
"""

from app.store.main import (
    SqliteStore,
    close_store,
    get_store,
)
from app.store.postgres import PostgresStore, postgres_schema_for_namespace
from app.store.search import cosine_similarity

__all__ = [
    "SqliteStore",
    "PostgresStore",
    "postgres_schema_for_namespace",
    "close_store",
    "get_store",
    "cosine_similarity",
]
