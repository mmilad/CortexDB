"""DB-writing workflow for deterministic logic ingest analysis."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from app.ingest import analyze_ingest
from app.ingest.rules import load_ingest_analysis_config
from app.processors.safe import process_text_safe
from app.processors.service import ProcessorService
from app.processors.validation import ProcessorValidationError, validate_processor_response
from app.schemas.dataset import DatasetRecord
from app.schemas.ingest_analysis import ExistingDatasetSummary
from app.schemas.processor import ProcessorRequest, ProcessorResponse
from app.schemas.session import (
    DerivedJobResult,
    IngestCanonicalEntityTrace,
    IngestRouteTargetTrace,
    IngestTrace,
)
from app.store import SqliteStore

_SESSION_MEMORY_DATASET_KEY = "session_memory"
_PRIMITIVES_DATASET_KEY = "ingest_primitives"
_CANDIDATES_DATASET_KEY = "ingest_candidates"
_ENTITY_KIND_MARKERS = {
    "alias",
    "entity",
    "framework",
    "library",
    "org",
    "organization",
    "person",
    "place",
    "product",
    "tool",
}
_NON_CANONICAL_KINDS = {
    "command",
    "constraint",
    "decision",
    "error",
    "event",
    "fact",
    "preference",
    "reference",
    "routing_hint",
    "task",
    "time",
}


def _digest(*parts: object, length: int = 20) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _slug(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:max_length] or "entity"


def _ensure_dataset(store: SqliteStore, key: str, *, kind: str) -> None:
    if store.get_dataset(key):
        return
    if kind == "session_memory":
        record = DatasetRecord(
            dataset_key=key,
            display_name="Session Memory",
            schema_version="v1",
            semantic_description="Logic-ingest chunks preserving session text for retrieval and audit.",
            usage_guidance="Use for recent or historical conversation chunks written by /ingest.",
            llm_summary="Session text chunks produced by deterministic logic ingest.",
            retrieval_capabilities=["keyword"],
            content_kind="custom",
            capability_tags=["session", "logic_ingest", "chunks"],
            entity_types=["SessionMemory"],
            access_patterns=["keyword_search", "autocontext"],
            filterable_fields=["source", "session_id", "raw_text_id", "session_message_id", "chunk_id"],
        )
    elif kind == "primitive":
        record = DatasetRecord(
            dataset_key=key,
            display_name="Ingest Primitives",
            schema_version="v1",
            semantic_description="Deterministic primitive extractions from logic ingest.",
            usage_guidance="Use for tasks, decisions, entities, aliases, time references, and custom rule-pack primitives.",
            llm_summary="Primitive memory items extracted by deterministic ingest rules.",
            retrieval_capabilities=["keyword"],
            content_kind="custom",
            capability_tags=["logic_ingest", "primitives"],
            entity_types=["Primitive"],
            access_patterns=["keyword_search", "by_kind", "by_session"],
            filterable_fields=[
                "source",
                "session_id",
                "raw_text_id",
                "session_message_id",
                "primitive_kind",
                "primitive_subkind",
            ],
        )
    else:
        record = DatasetRecord(
            dataset_key=key,
            display_name="Ingest Candidates",
            schema_version="v1",
            semantic_description="Unclassified or weakly classified observations collected during logic ingest.",
            usage_guidance="Use as evidence for external LLMs to propose new datasets and ingest rules after a turn.",
            llm_summary="Candidate observations for future dataset/rule evolution.",
            retrieval_capabilities=["keyword"],
            content_kind="custom",
            capability_tags=["logic_ingest", "candidates", "unclassified"],
            entity_types=["CandidateObservation"],
            access_patterns=["keyword_search", "by_label", "by_session"],
            filterable_fields=[
                "source",
                "session_id",
                "raw_text_id",
                "session_message_id",
                "candidate_label",
                "suggested_dataset_key",
            ],
        )
    store.upsert_dataset(key, record.model_dump())


def _ensure_canonical_dataset(store: SqliteStore, dataset_key: str, *, entity_kind: str) -> None:
    if store.get_dataset(dataset_key):
        return
    label = dataset_key.replace("_", " ").title()
    entity_label = entity_kind.replace("_", " ").title()
    record = DatasetRecord(
        dataset_key=dataset_key,
        display_name=label,
        schema_version="v1",
        semantic_description=f"Canonical {entity_label} knowledge created from ingest rules.",
        usage_guidance=(
            f"Use for canonical {entity_label} records with provenance links back to "
            "raw text, session messages, and source-bound observations."
        ),
        llm_summary=f"Canonical {entity_label} records maintained by logic ingest.",
        retrieval_capabilities=["keyword"],
        content_kind="custom",
        capability_tags=["canonical_entity", entity_kind, "logic_ingest"],
        entity_types=[entity_label],
        access_patterns=["keyword_search", "by_canonical_name", "by_evidence"],
        filterable_fields=["memory_role", "entity_kind", "canonical_name"],
    )
    store.upsert_dataset(dataset_key, record.model_dump())


def _existing_datasets(store: SqliteStore) -> list[ExistingDatasetSummary]:
    datasets: list[ExistingDatasetSummary] = []
    for key, row in store.list_datasets().items():
        payload = {"dataset_key": key, **row}
        try:
            datasets.append(ExistingDatasetSummary.model_validate(payload))
        except ValueError:
            continue
    return datasets


def _relationship_id(edge: dict[str, Any]) -> str:
    return "logic-edge-" + _digest(
        edge["source_type"],
        edge["source_key"],
        edge["edge_type"],
        edge["target_type"],
        edge["target_key"],
        edge.get("description", ""),
    )


def _remap_key(key: str, mapping: dict[str, str]) -> str:
    return mapping.get(key, key)


def _is_entity_like(kind: str, target_dataset_key: str | None) -> bool:
    if not target_dataset_key:
        return False
    normalized_kind = kind.casefold()
    if normalized_kind in _NON_CANONICAL_KINDS:
        return False
    parts = set(filter(None, re.split(r"[^a-z0-9]+", normalized_kind)))
    return bool(parts & _ENTITY_KIND_MARKERS)


def _canonical_entity_id(dataset_key: str, kind: str, text: str) -> str:
    slug = _slug(text)
    return f"canonical-{_slug(dataset_key, max_length=32)}-{_slug(kind, max_length=24)}-{slug}"


def _merge_unique(existing: list[Any], additions: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in [*existing, *additions]:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _increment(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    if amount <= 0:
        return
    mapping[key] = mapping.get(key, 0) + amount


def _upsert_canonical_entity(
    store: SqliteStore,
    *,
    dataset_key: str,
    entity_kind: str,
    text: str,
    observation_id: str,
    raw_text_id: str,
    session_id: str,
    confidence: float | None,
) -> str:
    _ensure_canonical_dataset(store, dataset_key, entity_kind=entity_kind)
    canonical_id = _canonical_entity_id(dataset_key, entity_kind, text)
    now = _now_iso()
    existing = store.get_memory_item(canonical_id)
    normalized = _normalized_text(text)
    if existing:
        metadata = dict(existing.get("metadata", {}))
        evidence_count = int(metadata.get("evidence_count", 0)) + 1
        confidences = [
            value for value in metadata.get("confidence_values", [])
            if isinstance(value, (int, float))
        ]
        if confidence is not None:
            confidences.append(confidence)
        metadata.update(
            {
                "memory_role": "canonical_entity",
                "entity_kind": entity_kind,
                "canonical_name": metadata.get("canonical_name") or text,
                "normalized_text": normalized,
                "evidence_count": evidence_count,
                "last_seen_at": now,
                "source_observation_ids": _merge_unique(
                    list(metadata.get("source_observation_ids", [])),
                    [observation_id],
                ),
                "source_raw_text_ids": _merge_unique(
                    list(metadata.get("source_raw_text_ids", [])),
                    [raw_text_id],
                ),
                "source_session_ids": _merge_unique(
                    list(metadata.get("source_session_ids", [])),
                    [session_id],
                ),
                "confidence_values": confidences,
                "confidence_avg": round(sum(confidences) / len(confidences), 6) if confidences else None,
            }
        )
        raw_text = existing.get("raw_text") or text
    else:
        metadata = {
            "memory_role": "canonical_entity",
            "source": "logic_ingest",
            "logic_ingest": True,
            "entity_kind": entity_kind,
            "canonical_name": text,
            "aliases": [text],
            "normalized_text": normalized,
            "evidence_count": 1,
            "first_seen_at": now,
            "last_seen_at": now,
            "source_observation_ids": [observation_id],
            "source_raw_text_ids": [raw_text_id],
            "source_session_ids": [session_id],
            "confidence_values": [confidence] if confidence is not None else [],
            "confidence_avg": confidence,
        }
        raw_text = text
    store.insert_memory_item(
        {
            "id": canonical_id,
            "dataset_key": dataset_key,
            "raw_text": raw_text,
            "metadata": metadata,
        }
    )
    return canonical_id


def _relationship(
    store: SqliteStore,
    *,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    edge_type: str,
    description: str,
) -> str:
    rel = {
        "source_type": source_type,
        "source_key": source_key,
        "target_type": target_type,
        "target_key": target_key,
        "edge_type": edge_type,
        "join_fields": [],
        "description": description,
    }
    rel["id"] = _relationship_id(rel)
    store.upsert_relationship(rel)
    return rel["id"]


def _processor_rules_payload(store: SqliteStore, namespace: str | None) -> list[dict[str, Any]]:
    return [
        {
            "key": row.get("key"),
            "semantic_rules": row.get("semantic_rules", []),
            "entity_hints": row.get("entity_hints", []),
            "primitive_rules": row.get("primitive_rules", []),
            "aliases": row.get("aliases", []),
            "routing_hints": row.get("routing_hints", []),
        }
        for row in store.list_ingest_rule_packs(namespace=namespace, active_only=True)
    ]


async def _process_text(
    *,
    text: str,
    processor_svc: ProcessorService | None,
    namespace: str | None,
    store: SqliteStore,
) -> tuple[ProcessorResponse, DerivedJobResult]:
    request = ProcessorRequest(
        text=text,
        strategy=processor_svc.strategy if processor_svc is not None else "safe",  # type: ignore[arg-type]
        max_chars=2000,
        overlap_chars=200,
        extract_primitives=True,
        classify=processor_svc.classify_enabled if processor_svc is not None else False,
        known_match_threshold=processor_svc.known_match_threshold if processor_svc is not None else 0.72,
        candidate_threshold=processor_svc.candidate_threshold if processor_svc is not None else 0.45,
        metadata={"namespace": namespace, "rules": _processor_rules_payload(store, namespace)},
    )
    if processor_svc is None or not processor_svc.is_enabled():
        response = process_text_safe(request.model_copy(update={"strategy": "safe"}))
        return response, DerivedJobResult(
            name="processor",
            status="skipped",
            detail="processor disabled; used deterministic fallback",
        )
    try:
        response = await processor_svc.process_text(request)
        response = validate_processor_response(text, response, max_chars=request.max_chars)
        return response, DerivedJobResult(
            name="processor",
            status="completed",
            detail=(
                f"provider={processor_svc.provider}; chunks={len(response.chunks)}; "
                f"entities={len(response.entities)}; classifications={len(response.classifications)}; "
                f"candidates={len(response.candidates)}"
            ),
        )
    except (Exception, ProcessorValidationError) as exc:
        if not processor_svc.graceful_fallback:
            raise
        response = process_text_safe(request.model_copy(update={"strategy": "safe"}))
        return response, DerivedJobResult(
            name="processor",
            status="skipped",
            detail=f"processor unavailable; used deterministic fallback: {exc}",
        )


def _insert_candidate(
    store: SqliteStore,
    *,
    raw_text: str,
    label: str,
    source: str,
    session_id: str,
    raw_text_id: str,
    session_message_id: str,
    metadata: dict[str, Any],
) -> str:
    item_id = f"candidate-{_digest(session_id, raw_text_id, source, label, raw_text)}"
    store.insert_memory_item(
        {
            "id": item_id,
            "dataset_key": _CANDIDATES_DATASET_KEY,
            "raw_text": raw_text,
            "metadata": {
                **metadata,
                "memory_role": "candidate",
                "source": source,
                "logic_ingest": True,
                "session_id": session_id,
                "raw_text_id": raw_text_id,
                "session_message_id": session_message_id,
                "candidate_label": label,
            },
        }
    )
    return item_id


def _persist_analysis_observations(
    *,
    store: SqliteStore,
    primitive_writes: list[Any],
    session_id: str,
    raw_text_id: str,
    session_message_id: str,
) -> tuple[
    list[str],
    dict[str, str],
    list[str],
    list[IngestCanonicalEntityTrace],
    dict[str, int],
]:
    item_ids: list[str] = []
    key_map: dict[str, str] = {}
    relationship_ids: list[str] = []
    canonical_entities: list[IngestCanonicalEntityTrace] = []
    observation_kinds: dict[str, int] = {}
    observation_item_kinds: dict[str, str] = {}
    entity_groups: dict[tuple[str, str, str], list[Any]] = {}

    for write in primitive_writes:
        target_dataset_key = write.metadata.get("target_dataset_key")
        if _is_entity_like(write.kind, target_dataset_key):
            key = (write.kind, _normalized_text(write.raw_text), target_dataset_key)
            entity_groups.setdefault(key, []).append(write)
            continue

        item_id = f"observation-{_digest(raw_text_id, write.item_id)}"
        metadata = {
            **write.metadata,
            "memory_role": "observation",
            "observation_scope": "source_bound",
            "source": "logic_ingest",
            "logic_ingest": True,
            "session_id": session_id,
            "raw_text_id": raw_text_id,
            "session_message_id": session_message_id,
            "mentions": [
                {
                    "primitive_id": write.metadata.get("primitive_id"),
                    "char_start": write.metadata.get("char_start"),
                    "char_end": write.metadata.get("char_end"),
                    "chunk_ids": write.metadata.get("chunk_ids", []),
                    "confidence": write.confidence,
                }
            ],
            "mention_count": 1,
        }
        store.insert_memory_item(
            {
                "id": item_id,
                "dataset_key": _PRIMITIVES_DATASET_KEY,
                "raw_text": write.raw_text,
                "metadata": metadata,
            }
        )
        item_ids.append(item_id)
        key_map[write.item_id] = item_id
        observation_item_kinds[item_id] = write.kind

    for (kind, _normalized, target_dataset_key), writes in entity_groups.items():
        first = writes[0]
        item_id = f"observation-{_digest(raw_text_id, kind, target_dataset_key, _normalized)}"
        mentions = [
            {
                "primitive_id": write.metadata.get("primitive_id"),
                "char_start": write.metadata.get("char_start"),
                "char_end": write.metadata.get("char_end"),
                "chunk_ids": write.metadata.get("chunk_ids", []),
                "confidence": write.confidence,
            }
            for write in writes
        ]
        chunk_ids: list[str] = []
        primitive_ids: list[str] = []
        confidence_values: list[float] = []
        for write in writes:
            primitive_id = write.metadata.get("primitive_id")
            if isinstance(primitive_id, str):
                primitive_ids.append(primitive_id)
            chunk_ids.extend([
                chunk_id for chunk_id in write.metadata.get("chunk_ids", [])
                if isinstance(chunk_id, str)
            ])
            confidence_values.append(write.confidence)
            key_map[write.item_id] = item_id

        metadata = {
            **first.metadata,
            "memory_role": "observation",
            "observation_scope": "source_bound",
            "observation_kind": "entity_mention",
            "source": "logic_ingest",
            "logic_ingest": True,
            "session_id": session_id,
            "raw_text_id": raw_text_id,
            "session_message_id": session_message_id,
            "primitive_kind": kind,
            "target_dataset_key": target_dataset_key,
            "normalized_text": _normalized,
            "mentions": mentions,
            "mention_count": len(mentions),
            "primitive_ids": _merge_unique([], primitive_ids),
            "chunk_ids": _merge_unique([], chunk_ids),
            "confidence_values": confidence_values,
            "confidence_avg": round(sum(confidence_values) / len(confidence_values), 6)
            if confidence_values else None,
        }
        store.insert_memory_item(
            {
                "id": item_id,
                "dataset_key": _PRIMITIVES_DATASET_KEY,
                "raw_text": first.raw_text,
                "metadata": metadata,
            }
        )
        item_ids.append(item_id)
        observation_item_kinds[item_id] = kind

        canonical_id = _upsert_canonical_entity(
            store,
            dataset_key=target_dataset_key,
            entity_kind=kind,
            text=first.raw_text,
            observation_id=item_id,
            raw_text_id=raw_text_id,
            session_id=session_id,
            confidence=metadata["confidence_avg"],
        )
        relationship_ids.append(
            _relationship(
                store,
                source_type="memory_item",
                source_key=item_id,
                target_type="memory_item",
                target_key=canonical_id,
                edge_type="shared_entity",
                description=f"Observation resolves to canonical {kind} entity {first.raw_text}.",
            )
        )
        relationship_ids.append(
            _relationship(
                store,
                source_type="memory_item",
                source_key=canonical_id,
                target_type="dataset",
                target_key=target_dataset_key,
                edge_type="related",
                description=f"Canonical {kind} entity belongs to dataset {target_dataset_key}.",
            )
        )
        canonical_entities.append(
            IngestCanonicalEntityTrace(
                id=canonical_id,
                dataset_key=target_dataset_key,
                entity_kind=kind,
                name=first.raw_text,
                observation_id=item_id,
                mention_count=len(mentions),
            )
        )

    for kind in observation_item_kinds.values():
        _increment(observation_kinds, kind)
    return _merge_unique([], item_ids), key_map, relationship_ids, canonical_entities, observation_kinds


def _persist_processor_observations(
    *,
    store: SqliteStore,
    response: ProcessorResponse,
    session_id: str,
    raw_text_id: str,
    session_message_id: str,
) -> tuple[list[str], list[str]]:
    primitive_ids: list[str] = []
    candidate_ids: list[str] = []

    for entity in response.entities:
        item_id = f"processor-entity-{_digest(session_id, entity.label, entity.char_start, entity.char_end, entity.text)}"
        store.insert_memory_item(
            {
                "id": item_id,
                "dataset_key": _PRIMITIVES_DATASET_KEY,
                "raw_text": entity.text,
                "metadata": {
                    **entity.metadata,
                    "memory_role": "observation",
                    "observation_scope": "source_bound",
                    "source": "processor_entity",
                    "logic_ingest": True,
                    "session_id": session_id,
                    "raw_text_id": raw_text_id,
                    "session_message_id": session_message_id,
                    "primitive_kind": "entity",
                    "primitive_subkind": entity.label,
                    "confidence": entity.confidence,
                    "char_start": entity.char_start,
                    "char_end": entity.char_end,
                },
            }
        )
        primitive_ids.append(item_id)

    for classification in response.classifications:
        metadata = {
            **classification.metadata,
            "classification_label": classification.label,
            "classification_score": classification.score,
            "matched_rule_key": classification.matched_rule_key,
            "target_dataset_key": classification.target_dataset_key,
            "char_start": classification.char_start,
            "char_end": classification.char_end,
        }
        if classification.target_dataset_key:
            item_id = f"processor-classification-{_digest(session_id, classification.label, classification.score, raw_text_id)}"
            store.insert_memory_item(
                {
                    "id": item_id,
                    "dataset_key": _PRIMITIVES_DATASET_KEY,
                    "raw_text": classification.label,
                    "metadata": {
                        **metadata,
                        "memory_role": "observation",
                        "observation_scope": "source_bound",
                        "source": "processor_classification",
                        "logic_ingest": True,
                        "session_id": session_id,
                        "raw_text_id": raw_text_id,
                        "session_message_id": session_message_id,
                        "primitive_kind": "classification",
                    },
                }
            )
            primitive_ids.append(item_id)
        else:
            candidate_ids.append(
                _insert_candidate(
                    store,
                    raw_text=classification.label,
                    label=classification.label,
                    source="processor_classification",
                    session_id=session_id,
                    raw_text_id=raw_text_id,
                    session_message_id=session_message_id,
                    metadata=metadata,
                )
            )

    for phrase in response.phrases:
        candidate_ids.append(
            _insert_candidate(
                store,
                raw_text=phrase.text,
                label=phrase.label,
                source="processor_phrase",
                session_id=session_id,
                raw_text_id=raw_text_id,
                session_message_id=session_message_id,
                metadata={
                    **phrase.metadata,
                    "score": phrase.score,
                    "char_start": phrase.char_start,
                    "char_end": phrase.char_end,
                },
            )
        )

    for candidate in response.candidates:
        candidate_ids.append(
            _insert_candidate(
                store,
                raw_text=candidate.text,
                label=candidate.label,
                source="processor_candidate",
                session_id=session_id,
                raw_text_id=raw_text_id,
                session_message_id=session_message_id,
                metadata={
                    "score": candidate.score,
                    "suggested_dataset_key": candidate.suggested_dataset_key,
                    "evidence": candidate.evidence,
                    "char_start": candidate.char_start,
                    "char_end": candidate.char_end,
                },
            )
        )

    return primitive_ids, candidate_ids


def _processor_observation_kinds(response: ProcessorResponse) -> dict[str, int]:
    kinds: dict[str, int] = {}
    if response.entities:
        _increment(kinds, "entity", len(response.entities))
    if response.classifications:
        _increment(
            kinds,
            "classification",
            sum(1 for item in response.classifications if item.target_dataset_key),
        )
    return kinds


def _candidate_labels_from_response(response: ProcessorResponse) -> list[str]:
    labels = [
        classification.label
        for classification in response.classifications
        if not classification.target_dataset_key
    ]
    labels.extend(phrase.label for phrase in response.phrases)
    labels.extend(candidate.label for candidate in response.candidates)
    return _merge_unique([], labels)


async def run_logic_ingest_workflow(
    *,
    store: SqliteStore,
    text: str,
    derive: bool,
    session_id: str,
    raw_text_id: str,
    session_message_id: str,
    namespace: str | None = None,
    processor_svc: ProcessorService | None = None,
) -> tuple[list[DerivedJobResult], IngestTrace]:
    if not derive:
        return (
            [DerivedJobResult(name="logic_analysis", status="skipped", detail="derive=false")],
            IngestTrace(status="skipped"),
        )

    _ensure_dataset(store, _SESSION_MEMORY_DATASET_KEY, kind="session_memory")
    _ensure_dataset(store, _PRIMITIVES_DATASET_KEY, kind="primitive")
    _ensure_dataset(store, _CANDIDATES_DATASET_KEY, kind="candidate")

    processor_response, processor_job = await _process_text(
        text=text,
        processor_svc=processor_svc,
        namespace=namespace,
        store=store,
    )

    config = load_ingest_analysis_config(store, namespace=namespace)
    datasets = _existing_datasets(store)
    result = analyze_ingest(
        text,
        session_id=session_id,
        config=config,
        existing_datasets=datasets,
    )

    key_map: dict[str, str] = {
        f"raw-{_digest(session_id, text, length=16)}": raw_text_id,
        f"msg-{_digest(session_id, text, length=16)}": session_message_id,
        session_id: session_id,
    }

    session_item_ids: list[str] = []
    for write in result.session_memory_writes:
        metadata = {
            **write.metadata,
            "memory_role": "provenance",
            "source": "logic_ingest",
            "logic_ingest": True,
            "session_id": session_id,
            "raw_text_id": raw_text_id,
            "session_message_id": session_message_id,
        }
        store.insert_memory_item(
            {
                "id": write.item_id,
                "dataset_key": write.dataset_key,
                "raw_text": write.raw_text,
                "metadata": metadata,
            }
        )
        session_item_ids.append(write.item_id)
        chunk_id = metadata.get("chunk_id")
        if isinstance(chunk_id, str):
            key_map[chunk_id] = write.item_id

    (
        primitive_item_ids,
        primitive_key_map,
        canonical_relationship_ids,
        canonical_entities,
        observation_kinds,
    ) = _persist_analysis_observations(
        store=store,
        primitive_writes=result.primitive_write_proposals,
        session_id=session_id,
        raw_text_id=raw_text_id,
        session_message_id=session_message_id,
    )
    key_map.update(primitive_key_map)

    processor_primitive_ids, candidate_item_ids = _persist_processor_observations(
        store=store,
        response=processor_response,
        session_id=session_id,
        raw_text_id=raw_text_id,
        session_message_id=session_message_id,
    )
    primitive_item_ids.extend(processor_primitive_ids)
    primitive_item_ids = _merge_unique([], primitive_item_ids)
    for kind, count in _processor_observation_kinds(processor_response).items():
        _increment(observation_kinds, kind, count)

    candidate_labels = _candidate_labels_from_response(processor_response)
    if not result.dataset_routes:
        for candidate in result.dataset_creation_candidates:
            candidate_item_ids.append(
                _insert_candidate(
                    store,
                    raw_text=candidate.label,
                    label=candidate.label,
                    source="logic_dataset_candidate",
                    session_id=session_id,
                    raw_text_id=raw_text_id,
                    session_message_id=session_message_id,
                    metadata={
                        "suggested_dataset_key": candidate.suggested_dataset_key,
                        "evidence_count": candidate.evidence_count,
                        "ready_to_create": candidate.ready_to_create,
                        "reasons": candidate.reasons,
                        "primitive_ids": candidate.primitive_ids,
                        "chunk_ids": candidate.chunk_ids,
                    },
                )
            )
            candidate_labels.append(candidate.label)

    relationship_ids: list[str] = []
    relationship_ids.extend(canonical_relationship_ids)
    for edge in result.graph_edges:
        rel = {
            "source_type": edge.source_type,
            "source_key": _remap_key(edge.source_key, key_map),
            "target_type": edge.target_type,
            "target_key": _remap_key(edge.target_key, key_map),
            "edge_type": edge.edge_type,
            "join_fields": [],
            "description": edge.description,
        }
        rel["id"] = _relationship_id(rel)
        store.upsert_relationship(rel)
        relationship_ids.append(rel["id"])
    relationship_ids = _merge_unique([], relationship_ids)

    trace = IngestTrace(
        status="completed",
        chunks_written=len(session_item_ids),
        observations_written=len(primitive_item_ids),
        canonical_entities_upserted=len(canonical_entities),
        candidate_observations_written=len(candidate_item_ids),
        graph_edges_written=len(relationship_ids),
        route_targets=[
            IngestRouteTargetTrace(
                dataset_key=route.dataset_key,
                score=route.score,
                reasons=route.reasons,
                primitive_count=len(route.primitive_ids),
                chunk_count=len(route.chunk_ids),
            )
            for route in result.dataset_routes
        ],
        observation_kinds=dict(sorted(observation_kinds.items())),
        canonical_entities=canonical_entities,
        candidate_labels=_merge_unique([], candidate_labels),
    )

    return (
        [
            processor_job,
            DerivedJobResult(
                name="logic_analysis",
                status="completed",
                detail=(
                    f"chunks={len(result.chunks)}; primitives={len(result.primitives)}; "
                    f"routes={len(result.dataset_routes)}; graph_edges={len(result.graph_edges)}"
                ),
                dataset_keys=[route.dataset_key for route in result.dataset_routes],
            ),
            DerivedJobResult(
                name="session_memory",
                status="completed" if session_item_ids else "skipped",
                detail=f"written_items={len(session_item_ids)}",
                dataset_keys=[_SESSION_MEMORY_DATASET_KEY] if session_item_ids else [],
                item_ids=session_item_ids,
            ),
            DerivedJobResult(
                name="primitive_memory",
                status="completed" if primitive_item_ids else "skipped",
                detail=f"written_items={len(primitive_item_ids)}",
                dataset_keys=[_PRIMITIVES_DATASET_KEY] if primitive_item_ids else [],
                item_ids=primitive_item_ids,
            ),
            DerivedJobResult(
                name="graph_edges",
                status="completed" if relationship_ids else "skipped",
                detail=f"written_relationships={len(relationship_ids)}",
                item_ids=relationship_ids,
            ),
            DerivedJobResult(
                name="candidate_observations",
                status="completed" if candidate_item_ids else "skipped",
                detail=f"written_items={len(candidate_item_ids)}",
                dataset_keys=[_CANDIDATES_DATASET_KEY] if candidate_item_ids else [],
                item_ids=candidate_item_ids,
            ),
        ],
        trace,
    )
