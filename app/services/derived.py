"""Derived ingest workflow.

The v1 implementation treats LLM-backed extraction as optional infrastructure:
raw/session writes are authoritative and never blocked by this module.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.llm import LLMService
from app.schemas.dataset import DatasetRecord
from app.schemas.session import DerivedJobResult
from app.store import SqliteStore

_DERIVED_TYPES = ("facts", "decisions", "goals", "knowledge")


def _dataset_key(kind: str) -> str:
    return f"derived_{kind}"


def _ensure_derived_dataset(store: SqliteStore, kind: str) -> str:
    key = _dataset_key(kind)
    if store.get_dataset(key):
        return key
    label = kind.replace("_", " ").title()
    record = DatasetRecord(
        dataset_key=key,
        display_name=f"Derived {label}",
        schema_version="v1",
        semantic_description=f"LLM-extracted {kind} from session-aware ingest.",
        usage_guidance=f"Use for compact retrieval of durable {kind} extracted by CortexDB ingest.",
        llm_summary=f"Small generated memory records containing durable {kind}.",
        retrieval_capabilities=["keyword"],
        content_kind="custom",
        capability_tags=["derived", kind],
        entity_types=[kind.rstrip("s").title()],
        access_patterns=["keyword_search", "autocontext"],
        filterable_fields=["source", "derived_kind", "session_id"],
    )
    store.upsert_dataset(key, record.model_dump())
    return key


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return ""


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("metadata"), dict):
        return dict(value["metadata"])
    return {}


def _score(value: Any) -> float | None:
    if isinstance(value, dict) and value.get("score") is not None:
        try:
            return float(value["score"])
        except (TypeError, ValueError):
            return None
    return None


def run_derived_workflow(
    *,
    store: SqliteStore,
    text: str,
    dataset_policy: str,
    dataset_keys: list[str],
    derive: bool,
    llm_svc: LLMService | None = None,
    session_id: str | None = None,
    raw_text_id: str | None = None,
) -> list[DerivedJobResult]:
    if not derive:
        return [DerivedJobResult(name="derived", status="skipped", detail="derive=false")]

    if llm_svc is None or not llm_svc.is_enabled():
        return [
            DerivedJobResult(
                name="llm_extraction",
                status="skipped",
                detail=(
                    "No CORTEXDB_LLM_PROVIDER=api configuration is available. "
                    "Session and raw text were stored; derived extraction was skipped."
                ),
            ),
            DerivedJobResult(
                name="dataset_routing",
                status="skipped",
                detail=f"policy={dataset_policy}; explicit_dataset_keys={dataset_keys}",
                dataset_keys=dataset_keys,
            ),
        ]

    try:
        extracted = llm_svc.extract_memory(text)
    except Exception as exc:
        return [
            DerivedJobResult(
                name="llm_extraction",
                status="failed",
                detail=str(exc),
            )
        ]

    item_ids: list[str] = []
    written_datasets: list[str] = []
    for kind in _DERIVED_TYPES:
        records = extracted.get(kind, [])
        if not records:
            continue
        if dataset_policy == "never_create":
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        elif dataset_policy == "explicit_only":
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        elif dataset_keys:
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        else:
            target_keys = [_ensure_derived_dataset(store, kind)]

        for dataset_key in target_keys:
            if dataset_key not in written_datasets:
                written_datasets.append(dataset_key)
            for record in records:
                raw = _normalize_text(record)
                if not raw:
                    continue
                item_id = f"derived-{kind}-{uuid.uuid4().hex}"
                metadata = {
                    **_normalize_metadata(record),
                    "derived_kind": kind,
                    "source": "llm_extraction",
                    "session_id": session_id,
                    "raw_text_id": raw_text_id,
                    "score": _score(record),
                }
                store.insert_memory_item({
                    "id": item_id,
                    "dataset_key": dataset_key,
                    "raw_text": raw,
                    "metadata": metadata,
                })
                item_ids.append(item_id)

    status = "completed" if item_ids else "skipped"
    detail = f"extracted_items={len(item_ids)}"
    if not item_ids:
        detail = "LLM returned no durable facts, decisions, goals, or knowledge."
    return [
        DerivedJobResult(
            name="llm_extraction",
            status=status,
            detail=detail,
            dataset_keys=written_datasets,
            item_ids=item_ids,
        )
    ]
