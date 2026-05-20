"""High-level session-aware ingest front door."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.processors import ProcessorService, get_processor_service
from app.schemas.session import DerivedJobResult, IngestRequest, IngestResult
from app.services.logic_ingest import run_logic_ingest_workflow
from app.services.session import (
    ensure_session,
    maybe_compact_session,
    write_raw_text,
    write_session_message,
)
from app.store import SqliteStore, get_store

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResult,
    summary="Session-aware ingest front door",
    description=(
        "Always stores both chat/session history and an auditable raw_text record. "
        "Derived LLM extraction and dataset routing are reported separately and do "
        "not block the durable writes."
    ),
)
async def ingest(
    body: IngestRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
    processor_svc: Annotated[ProcessorService, Depends(get_processor_service)],
) -> IngestResult:
    session = ensure_session(
        store,
        session_id=body.session_id or "main",
        session_type=body.session_type,
        scope_mode=body.scope_mode,
        namespace=body.namespace,
        dataset_policy=body.dataset_policy,
        metadata={"created_by": "ingest", **body.metadata.get("session", {})}
        if isinstance(body.metadata.get("session"), dict)
        else {"created_by": "ingest"},
    )
    raw = write_raw_text(
        store,
        text=body.text,
        source=body.source,
        relations={
            **body.raw_relations,
            "session_id": session.id,
        },
        metadata=body.metadata,
    )
    message = write_session_message(
        store,
        session_id=session.id,
        role=body.role,
        content=body.text,
        raw_text_id=raw.id,
        metadata=body.metadata,
    )

    summary = maybe_compact_session(
        store,
        session_id=session.id,
        max_context_tokens=body.max_context_tokens,
        summary_target_tokens=body.summary_target_tokens,
    )

    derived, trace = await run_logic_ingest_workflow(
        store=store,
        text=body.text,
        derive=body.derive,
        session_id=session.id,
        raw_text_id=raw.id,
        session_message_id=message.id,
        namespace=body.namespace,
        processor_svc=processor_svc,
    )
    if summary is not None:
        derived.insert(
            0,
            DerivedJobResult(
                name="session_summary",
                status="completed",
                detail=f"Compacted {len(summary.message_ids)} messages into {summary.id}.",
            ),
        )

    return IngestResult(
        session=session,
        message=message,
        raw_text=raw,
        derived=derived,
        trace=trace,
    )
