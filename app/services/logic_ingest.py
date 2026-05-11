"""DB-writing workflow for deterministic logic ingest analysis."""

from __future__ import annotations

import hashlib
from typing import Any

from app.ingest import analyze_ingest
from app.ingest.rules import load_ingest_analysis_config
from app.schemas.dataset import DatasetRecord
from app.schemas.ingest_analysis import ExistingDatasetSummary
from app.schemas.session import DerivedJobResult
from app.store import SqliteStore

_SESSION_MEMORY_DATASET_KEY = "session_memory"
_PRIMITIVES_DATASET_KEY = "ingest_primitives"


def _digest(*parts: object, length: int = 20) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


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
    else:
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
    store.upsert_dataset(key, record.model_dump())


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


def run_logic_ingest_workflow(
    *,
    store: SqliteStore,
    text: str,
    derive: bool,
    session_id: str,
    raw_text_id: str,
    session_message_id: str,
    namespace: str | None = None,
) -> list[DerivedJobResult]:
    if not derive:
        return [DerivedJobResult(name="logic_analysis", status="skipped", detail="derive=false")]

    _ensure_dataset(store, _SESSION_MEMORY_DATASET_KEY, kind="session_memory")
    _ensure_dataset(store, _PRIMITIVES_DATASET_KEY, kind="primitive")

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

    primitive_item_ids: list[str] = []
    for write in result.primitive_write_proposals:
        metadata = {
            **write.metadata,
            "source": "logic_ingest",
            "logic_ingest": True,
            "session_id": session_id,
            "raw_text_id": raw_text_id,
            "session_message_id": session_message_id,
        }
        store.insert_memory_item(
            {
                "id": write.item_id,
                "dataset_key": _PRIMITIVES_DATASET_KEY,
                "raw_text": write.raw_text,
                "metadata": metadata,
            }
        )
        primitive_item_ids.append(write.item_id)

    relationship_ids: list[str] = []
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

    return [
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
    ]
