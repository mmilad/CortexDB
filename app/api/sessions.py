"""Session history endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.session import SessionMessageRecord, SessionRecord, SessionUpdateRequest
from app.services.session import ensure_session
from app.store import SqliteStore, get_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionRecord], summary="List sessions")
def list_sessions(store: Annotated[SqliteStore, Depends(get_store)]) -> list[SessionRecord]:
    store.ensure_session("main")
    return [SessionRecord(**row) for row in store.list_sessions()]


@router.get("/{session_id}", response_model=SessionRecord, summary="Get one session")
def get_session(
    session_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> SessionRecord:
    if session_id == "main":
        return ensure_session(store)
    row = store.get_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionRecord(**row)


@router.patch("/{session_id}", response_model=SessionRecord, summary="Update or rename a session")
def update_session(
    session_id: str,
    body: SessionUpdateRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> SessionRecord:
    existing = ensure_session(store) if session_id == "main" else store.get_session(session_id)
    if not existing:
        raise HTTPException(status_code=404, detail="session not found")
    if body.id is not None and body.id != session_id and store.get_session(body.id):
        raise HTTPException(status_code=409, detail="target session already exists")

    metadata = dict(existing.get("metadata", {}))
    if body.metadata is not None:
        metadata.update(body.metadata)
    if body.title is not None:
        metadata["title"] = body.title

    updated = store.update_session(
        session_id,
        type=body.type,
        scope_mode=body.scope_mode,
        namespace=body.namespace,
        dataset_policy=body.dataset_policy,
        metadata=metadata,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="session not found")

    if body.id is not None and body.id != session_id:
        renamed = store.rename_session(session_id, body.id)
        if not renamed:
            raise HTTPException(status_code=409, detail="target session already exists")
        updated = renamed

    return SessionRecord(**updated)


@router.delete("/{session_id}", status_code=status.HTTP_200_OK, summary="Delete a session")
def delete_session(
    session_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    delete_related_chunks: bool = Query(
        default=False,
        description="When true, also delete raw texts and memory chunks/observations tied to this session.",
    ),
) -> dict[str, str | bool]:
    if not store.delete_session(session_id, delete_related_chunks=delete_related_chunks):
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": session_id, "delete_related_chunks": delete_related_chunks}


@router.get(
    "/{session_id}/history",
    response_model=list[SessionMessageRecord],
    summary="Get session chat history",
)
def get_session_history(
    session_id: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    autocontext_only: bool = Query(default=False),
) -> list[SessionMessageRecord]:
    if session_id == "main":
        store.ensure_session("main")
    elif not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    rows = store.list_session_messages(
        session_id,
        limit=limit,
        offset=offset,
        autocontext_only=autocontext_only,
    )
    return [SessionMessageRecord(**row) for row in rows]
