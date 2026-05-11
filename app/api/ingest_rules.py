from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.ingest import analyze_ingest
from app.ingest.rules import build_rule_guidance, load_ingest_analysis_config, validate_rule_pack
from app.schemas.ingest_analysis import ExistingDatasetSummary, IngestAnalysisRequest, IngestAnalysisResult
from app.schemas.ingest_rules import (
    IngestRuleGuidance,
    IngestRulePackRecord,
    IngestRulePackValidationResult,
)
from app.store import SqliteStore, get_store

router = APIRouter(prefix="/ingest", tags=["ingest-rules"])


@router.get(
    "/rule-packs/context",
    response_model=IngestRuleGuidance,
    summary="LLM guidance for proposing ingest rule packs",
    description=(
        "Returns schema and workflow guidance for LLM clients that want to teach "
        "CortexDB new deterministic ingest knowledge types without code changes."
    ),
)
def get_ingest_rule_context(
    store: Annotated[SqliteStore, Depends(get_store)],
    namespace: str | None = None,
) -> IngestRuleGuidance:
    return build_rule_guidance(store, namespace=namespace)


@router.post(
    "/rule-packs/validate",
    response_model=IngestRulePackValidationResult,
    summary="Validate an ingest rule pack without storing it",
)
def validate_ingest_rule_pack(
    pack: IngestRulePackRecord,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> IngestRulePackValidationResult:
    dataset_keys = set(store.list_datasets().keys())
    return validate_rule_pack(pack, known_dataset_keys=dataset_keys)


@router.post(
    "/rule-packs",
    response_model=IngestRulePackRecord,
    summary="Create or update an ingest rule pack",
    description="Validates and persists a schema-driven ingest rule pack as SQLite configuration.",
)
def upsert_ingest_rule_pack(
    pack: IngestRulePackRecord,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> IngestRulePackRecord:
    result = validate_rule_pack(pack, known_dataset_keys=set(store.list_datasets().keys()))
    if not result.accepted:
        raise HTTPException(status_code=422, detail={"errors": result.errors, "warnings": result.warnings})
    store.upsert_ingest_rule_pack(pack.key, pack.model_dump(mode="json"), namespace=pack.namespace)
    stored = store.get_ingest_rule_pack(pack.key, namespace=pack.namespace)
    return IngestRulePackRecord.model_validate(stored)


@router.get(
    "/rule-packs",
    response_model=list[IngestRulePackRecord],
    summary="List persisted ingest rule packs",
)
def list_ingest_rule_packs(
    store: Annotated[SqliteStore, Depends(get_store)],
    namespace: str | None = None,
    active_only: bool = False,
) -> list[IngestRulePackRecord]:
    return [
        IngestRulePackRecord.model_validate(row)
        for row in store.list_ingest_rule_packs(namespace=namespace, active_only=active_only)
    ]


@router.get(
    "/rule-packs/{key}",
    response_model=IngestRulePackRecord,
    summary="Get one persisted ingest rule pack",
)
def get_ingest_rule_pack(
    key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    namespace: str | None = None,
) -> IngestRulePackRecord:
    row = store.get_ingest_rule_pack(key, namespace=namespace)
    if not row:
        raise HTTPException(status_code=404, detail="ingest rule pack not found")
    return IngestRulePackRecord.model_validate(row)


@router.delete(
    "/rule-packs/{key}",
    summary="Delete one persisted ingest rule pack",
)
def delete_ingest_rule_pack(
    key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
    namespace: str | None = None,
) -> dict[str, str]:
    if not store.delete_ingest_rule_pack(key, namespace=namespace):
        raise HTTPException(status_code=404, detail="ingest rule pack not found")
    return {"deleted": key}


@router.post(
    "/analyze",
    response_model=IngestAnalysisResult,
    summary="Analyze ingest text using active persisted rule packs",
    description=(
        "Proposal-only analyzer endpoint. It does not mutate memory, does not call "
        "an LLM, and merges active persisted rule packs into the supplied analysis config."
    ),
)
def analyze_ingest_with_rule_packs(
    body: IngestAnalysisRequest,
    store: Annotated[SqliteStore, Depends(get_store)],
    namespace: str | None = None,
) -> IngestAnalysisResult:
    config = load_ingest_analysis_config(store, namespace=namespace, base_config=body.config)
    existing_datasets = body.existing_datasets
    if not existing_datasets:
        existing_datasets = [
            ExistingDatasetSummary.model_validate(row)
            for row in store.list_datasets().values()
        ]
    result = analyze_ingest(
        body.text,
        session_id=body.session_id,
        config=config,
        existing_datasets=existing_datasets,
        candidate_state=body.candidate_state,
    )
    result.metadata["active_rule_pack_count"] = len(store.list_ingest_rule_packs(namespace=namespace, active_only=True))
    return result
