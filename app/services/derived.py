"""Derived ingest workflow.

The v1 implementation treats LLM-backed extraction as optional infrastructure:
raw/session writes are authoritative and never blocked by this module.
"""

from __future__ import annotations

import uuid
import re
from typing import Any

from app.llm import LLMService
from app.schemas.dataset import DatasetRecord
from app.schemas.session import DerivedJobResult
from app.store import SqliteStore

_DERIVED_TYPES = ("facts", "decisions", "goals", "knowledge")
_DEFAULT_SCHEMA_VERSION = "cortexdb.derived_memory.v1"
_DATASET_KEY_RE = re.compile(r"[^a-z0-9_]+")


def _dataset_key(kind: str) -> str:
    return f"derived_{kind}"


def _safe_dataset_key(value: Any, fallback_kind: str) -> str:
    raw = str(value or "").strip().lower()
    raw = _DATASET_KEY_RE.sub("_", raw).strip("_")
    if not raw:
        raw = _dataset_key(fallback_kind)
    if not raw.startswith("derived_"):
        raw = f"derived_{raw}"
    return raw[:80]


def _ensure_derived_dataset(
    store: SqliteStore,
    kind: str,
    *,
    dataset_key: str | None = None,
    dataset: dict[str, Any] | None = None,
) -> str:
    key = _safe_dataset_key(dataset_key, kind)
    if store.get_dataset(key):
        return key
    dataset = dataset or {}
    label = kind.replace("_", " ").title()
    record = DatasetRecord(
        dataset_key=key,
        display_name=str(dataset.get("display_name") or f"Derived {label}")[:120],
        schema_version="v1",
        semantic_description=str(
            dataset.get("semantic_description")
            or f"LLM-extracted {kind} from session-aware ingest."
        ),
        usage_guidance=str(
            dataset.get("usage_guidance")
            or f"Use for compact retrieval of durable {kind} extracted by CortexDB ingest."
        ),
        llm_summary=str(
            dataset.get("llm_summary")
            or f"Small generated memory records containing durable {kind}."
        ),
        retrieval_capabilities=["keyword"],
        content_kind="custom",
        capability_tags=list(dict.fromkeys(["derived", kind, *list(dataset.get("capability_tags", []))])),
        entity_types=list(dataset.get("entity_types", [kind.rstrip("s").title()])),
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


def _normalize_kind(value: Any) -> str:
    raw = str(value or "custom").strip().lower()
    raw = _DATASET_KEY_RE.sub("_", raw).strip("_")
    return raw or "custom"


def _memory_records(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    memories = extracted.get("memories")
    if isinstance(memories, list):
        normalized = [m for m in memories if isinstance(m, dict)]
        if normalized:
            return normalized

    # Backward compatibility for the original facts/decisions/goals/knowledge
    # bucket format.
    records: list[dict[str, Any]] = []
    for kind in _DERIVED_TYPES:
        values = extracted.get(kind, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                records.append({
                    **value,
                    "kind": kind,
                    "dataset_key": value.get("dataset_key") or _dataset_key(kind),
                })
            elif isinstance(value, str):
                records.append({
                    "kind": kind,
                    "dataset_key": _dataset_key(kind),
                    "text": value,
                })
    return records


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

    memories = _memory_records(extracted)
    item_ids: list[str] = []
    written_datasets: list[str] = []
    for record in memories:
        kind = _normalize_kind(record.get("kind"))
        raw = _normalize_text(record)
        if not raw:
            continue
        if dataset_policy == "never_create":
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        elif dataset_policy == "explicit_only":
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        elif dataset_keys:
            target_keys = [key for key in dataset_keys if store.get_dataset(key)]
        else:
            target_keys = [
                _ensure_derived_dataset(
                    store,
                    kind,
                    dataset_key=record.get("dataset_key"),
                    dataset=record.get("dataset") if isinstance(record.get("dataset"), dict) else None,
                )
            ]

        for dataset_key in target_keys:
            if dataset_key not in written_datasets:
                written_datasets.append(dataset_key)
            item_id = f"derived-{kind}-{uuid.uuid4().hex}"
            metadata = {
                **_normalize_metadata(record),
                "schema_version": extracted.get("schema_version", _DEFAULT_SCHEMA_VERSION),
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
