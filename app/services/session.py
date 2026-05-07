"""Session and raw-text helpers for high-level CortexDB ingest."""

from __future__ import annotations

import uuid
from typing import Any

from app.schemas.session import (
    RawTextRecord,
    SessionMessageRecord,
    SessionRecord,
    SessionSummaryRecord,
)
from app.store import SqliteStore


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate used for context-window decisions."""
    return max(1, (len(text) + 3) // 4)


def ensure_session(
    store: SqliteStore,
    *,
    session_id: str = "main",
    session_type: str = "chat",
    scope_mode: str = "namespace",
    namespace: str | None = None,
    dataset_policy: str = "create_if_needed",
    metadata: dict[str, Any] | None = None,
) -> SessionRecord:
    row = store.ensure_session(
        session_id,
        type=session_type,
        scope_mode=scope_mode,
        namespace=namespace,
        dataset_policy=dataset_policy,
        metadata=metadata or {},
    )
    return SessionRecord(**row)


def write_raw_text(
    store: SqliteStore,
    *,
    text: str,
    source: str,
    relations: dict[str, Any] | None = None,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> RawTextRecord:
    raw_id = f"raw-{uuid.uuid4().hex}"
    store.insert_raw_text({
        "id": raw_id,
        "text": text,
        "source": source,
        "relations": relations or {},
        "score": score,
        "metadata": metadata or {},
    })
    saved = store.get_raw_text(raw_id)
    if saved is None:
        raise RuntimeError(f"could not create raw text '{raw_id}'")
    return RawTextRecord(**saved)


def write_session_message(
    store: SqliteStore,
    *,
    session_id: str,
    role: str,
    content: str,
    raw_text_id: str,
    metadata: dict[str, Any] | None = None,
) -> SessionMessageRecord:
    message_id = f"msg-{uuid.uuid4().hex}"
    store.insert_session_message({
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "raw_text_id": raw_text_id,
        "token_estimate": estimate_tokens(content),
        "autocontext_enabled": True,
        "metadata": metadata or {},
    })
    messages = store.list_session_messages(session_id, limit=10_000)
    for message in messages:
        if message["id"] == message_id:
            return SessionMessageRecord(**message)
    raise RuntimeError(f"could not create session message '{message_id}'")


def maybe_compact_session(
    store: SqliteStore,
    *,
    session_id: str,
    max_context_tokens: int,
    summary_target_tokens: int,
) -> SessionSummaryRecord | None:
    messages = store.list_session_messages(session_id, limit=10_000, autocontext_only=True)
    total = sum(int(m["token_estimate"]) for m in messages)
    if total <= max_context_tokens or len(messages) < 2:
        return None

    keep: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    running = 0
    for message in reversed(messages):
        if running < max_context_tokens // 2:
            keep.append(message)
            running += int(message["token_estimate"])
        else:
            compact.append(message)
    compact.reverse()
    if not compact:
        return None

    text = "\n".join(f"{m['role']}: {m['content']}" for m in compact)
    max_chars = max(200, summary_target_tokens * 4)
    summary_text = text[:max_chars]
    if len(text) > max_chars:
        summary_text = summary_text.rstrip() + "..."

    summary_id = f"summary-{uuid.uuid4().hex}"
    message_ids = [m["id"] for m in compact]
    store.insert_session_summary({
        "id": summary_id,
        "session_id": session_id,
        "summary": summary_text,
        "message_ids": message_ids,
        "token_estimate": estimate_tokens(summary_text),
        "metadata": {"strategy": "extractive_window"},
    })
    store.disable_session_messages_for_autocontext(session_id, message_ids, summary_id)
    summaries = store.list_session_summaries(session_id)
    for summary in summaries:
        if summary["id"] == summary_id:
            return SessionSummaryRecord(**summary)
    return None
