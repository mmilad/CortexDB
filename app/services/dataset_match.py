"""Deterministic dataset discovery with optional vector similarity boost.

Scoring when embedding is available:
  final_score = 0.6 * vector_score        (cosine similarity, 0-1 range → scaled ×20)
              + deterministic_score        (capabilities, tags, token overlap)

Scoring without embedding (fallback, v1 behaviour):
  final_score = deterministic_score only

The deterministic component ensures capability hard-filters still apply
and prevents semantically irrelevant but vector-adjacent datasets from
bubbling to the top.
"""

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
from app.store import SqliteStore, cosine_similarity

_VECTOR_WEIGHT = 20.0  # scale cosine (0-1) to be comparable to deterministic scores


def _tokens(text: str) -> set[str]:
    normalized = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return {t for t in normalized.split() if len(t) > 2}


def _intent_slug(intent: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", intent.lower()).strip("_")
    base = base[:40] if base else "dataset"
    return f"{base}_{uuid.uuid4().hex[:8]}"


def _deterministic_score(
    dataset: DatasetRecord,
    intent: str,
    required_capabilities: list[str],
    content_kind: str | None,
    tag_filters: list[str],
) -> tuple[float, list[str]]:
    """Capability + tag + token-overlap scoring (no vectors)."""
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
        if dataset.llm_summary:
            blob += f" {dataset.llm_summary}"
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
    store: SqliteStore,
    intent_vector: list[float] | None = None,
) -> DatasetDiscoverResponse:
    """Discover matching datasets with optional vector similarity boost.

    Parameters
    ----------
    request:       Discovery request with intent and filters.
    store:         SqliteStore instance.
    intent_vector: Pre-computed embedding of request.intent.
                   Pass None to use deterministic-only scoring (v1 fallback).
    """
    datasets_raw = store.list_datasets()
    loaded = [DatasetRecord(**d) for d in datasets_raw.values()]

    # Load stored embeddings if we have an intent vector
    dataset_vectors: dict[str, list[float]] = {}
    if intent_vector is not None:
        for row in store.list_datasets_with_embeddings():
            dataset_vectors[row["dataset_key"]] = row["embedding"]

    candidates: list[DatasetCandidate] = []

    for ds in loaded:
        if not _meets_requirements(ds, request.required_capabilities):
            continue

        det_score, reasons = _deterministic_score(
            ds,
            request.intent,
            request.required_capabilities,
            request.content_kind,
            request.tag_filters,
        )

        # Abort early if hard capability filter failed
        if any(r.startswith("missing_capabilities:") for r in reasons):
            continue

        total_score = det_score

        # Blend in vector score if available for this dataset
        if intent_vector is not None and ds.dataset_key in dataset_vectors:
            sim = cosine_similarity(intent_vector, dataset_vectors[ds.dataset_key])
            vec_contrib = sim * _VECTOR_WEIGHT
            total_score += vec_contrib
            reasons.append(f"vector_similarity:{sim:.4f}")

        candidates.append(DatasetCandidate(dataset=ds, score=total_score, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)

    score_threshold = 3.0 if request.required_capabilities else 1.0
    top = candidates[0] if candidates else None

    def _has_relevance_signals(reasons: list[str]) -> bool:
        return any(
            r.startswith("text_token_overlap:")
            or r.startswith("tag_match:")
            or r.startswith("vector_similarity:")
            or r == "content_kind_match"
            for r in reasons
        )

    use_existing = False
    if top is not None and top.score >= score_threshold:
        if request.required_capabilities:
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


# Kept for backward-compat with any direct callers of the old signature
def score_dataset(
    dataset: DatasetRecord,
    intent: str,
    required_capabilities: list[str],
    content_kind: str | None,
    tag_filters: list[str],
) -> tuple[float, list[str]]:
    return _deterministic_score(dataset, intent, required_capabilities, content_kind, tag_filters)
