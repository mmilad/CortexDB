"""Deterministic dataset discovery (no internal LLM or embedding similarity in v1)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.schemas.dataset import ContentKind, DatasetRecord
from app.schemas.discovery import (
    DatasetBlueprint,
    DatasetCandidate,
    DatasetDiscoverRequest,
    DatasetDiscoverResponse,
)


def _tokens(text: str) -> set[str]:
    normalized = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return {t for t in normalized.split() if len(t) > 2}


def _intent_slug(intent: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", intent.lower()).strip("_")
    base = base[:40] if base else "dataset"
    return f"{base}_{uuid.uuid4().hex[:8]}"


def score_dataset(
    dataset: DatasetRecord,
    intent: str,
    required_capabilities: list[str],
    content_kind: str | None,
    tag_filters: list[str],
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    caps = set(dataset.retrieval_capabilities)
    if required_capabilities:
        missing = [c for c in required_capabilities if c not in caps]
        if missing:
            reasons.append(f"missing_capabilities:{','.join(missing)}")
        else:
            score += 15.0
            reasons.append("required_capabilities_met")

    if content_kind and dataset.content_kind == content_kind:
        score += 8.0
        reasons.append("content_kind_match")

    intent_toks = _tokens(intent)
    if intent_toks:
        blob = f"{dataset.semantic_description} {dataset.usage_guidance}"
        text_toks = _tokens(blob)
        overlap = len(intent_toks & text_toks)
        if overlap:
            w = overlap * 0.8
            score += w
            reasons.append(f"text_token_overlap:{overlap}")

    if tag_filters:
        tags_lower = {t.lower() for t in dataset.capability_tags}
        hits = [t for t in tag_filters if t.lower() in tags_lower]
        if hits:
            score += 3.0 * len(hits)
            reasons.append(f"tag_match:{','.join(hits)}")

    return score, reasons


def _meets_requirements(dataset: DatasetRecord, required: list[str]) -> bool:
    if not required:
        return True
    caps = set(dataset.retrieval_capabilities)
    return all(c in caps for c in required)


def _parse_content_kind(value: str | None) -> ContentKind:
    if value == "documents":
        return "documents"
    if value == "events":
        return "events"
    return "custom"


def discover_datasets(
    request: DatasetDiscoverRequest,
    datasets: dict[str, dict[str, Any]],
) -> DatasetDiscoverResponse:
    loaded = [DatasetRecord(**d) for d in datasets.values()]
    candidates: list[DatasetCandidate] = []

    for ds in loaded:
        if not _meets_requirements(ds, request.required_capabilities):
            continue
        s, reasons = score_dataset(
            ds,
            request.intent,
            request.required_capabilities,
            request.content_kind,
            request.tag_filters,
        )
        candidates.append(DatasetCandidate(dataset=ds, score=s, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)

    score_threshold = 3.0 if request.required_capabilities else 1.0
    top = candidates[0] if candidates else None

    def _has_relevance_signals(reasons: list[str]) -> bool:
        return any(
            r.startswith("text_token_overlap:")
            or r.startswith("tag_match:")
            or r == "content_kind_match"
            for r in reasons
        )

    use_existing = False
    if top is not None and top.score >= score_threshold:
        if request.required_capabilities:
            # Capability match alone is weak; require intent overlap, tags, or kind.
            use_existing = _has_relevance_signals(top.reasons)
        else:
            use_existing = True

    blueprint: DatasetBlueprint | None = None
    if not use_existing:
        key = _intent_slug(request.intent)
        caps = list(dict.fromkeys(request.required_capabilities))
        if not caps:
            caps = ["filter_only"]
        ck = _parse_content_kind(request.content_kind)
        record = DatasetRecord(
            dataset_key=key,
            display_name=key.replace("_", " ").title()[:80],
            schema_version="v1",
            semantic_description=request.intent,
            usage_guidance=(
                "Created from discovery when no strong registry match was found. "
                "Refine semantic_description and usage_guidance after creation."
            ),
            retrieval_capabilities=caps,
            content_kind=ck,
            capability_tags=list(request.tag_filters),
        )
        blueprint = DatasetBlueprint(suggested_dataset_key=key, record=record)

    return DatasetDiscoverResponse(
        candidates=candidates[:20],
        recommended_action="use_existing" if use_existing else "create_new",
        suggested_blueprint=blueprint,
    )
