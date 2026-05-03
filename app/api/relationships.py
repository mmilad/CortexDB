"""CRUD API for typed, machine-traversable relationships between datasets and tools."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.relationship import RelationshipRecord
from app.store import SqliteStore, get_store

router = APIRouter(prefix="/relationships", tags=["relationships"])


@router.post(
    "",
    response_model=RelationshipRecord,
    summary="Create or update a relationship",
    description=(
        "Declare a typed edge between two registry nodes (datasets or tools). "
        "If 'id' is omitted a UUID is generated. "
        "Idempotent: re-posting the same id updates the record."
    ),
)
def upsert_relationship(
    record: RelationshipRecord,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> RelationshipRecord:
    if not record.id:
        record = record.model_copy(update={"id": str(uuid.uuid4())})
    store.upsert_relationship(record.model_dump())
    saved = store.get_relationship(record.id)
    return RelationshipRecord(**saved)


@router.get(
    "",
    response_model=list[RelationshipRecord],
    summary="List relationships",
    description=(
        "Return all relationships, or filter by node key. "
        "source_key and target_key both search in either direction."
    ),
)
def list_relationships(
    store: Annotated[SqliteStore, Depends(get_store)],
    node_key: str | None = Query(
        default=None,
        description="Return edges where this key appears as source OR target.",
    ),
) -> list[RelationshipRecord]:
    rows = store.list_relationships(source_key=node_key)
    return [RelationshipRecord(**r) for r in rows]


@router.get(
    "/{rel_id}",
    response_model=RelationshipRecord,
    summary="Get a specific relationship",
)
def get_relationship(
    rel_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> RelationshipRecord:
    row = store.get_relationship(rel_id)
    if not row:
        raise HTTPException(status_code=404, detail="relationship not found")
    return RelationshipRecord(**row)


@router.delete(
    "/{rel_id}",
    summary="Delete a relationship",
)
def delete_relationship(
    rel_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict:
    deleted = store.delete_relationship(rel_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="relationship not found")
    return {"deleted": rel_id}
