"""Session history endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.session import SessionMessageRecord, SessionRecord
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
