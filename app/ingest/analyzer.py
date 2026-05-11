"""Prototype logic-only ingest analysis.

This module is intentionally proposal-only: it does not write to the store,
does not call an LLM, and only uses embeddings when a caller injects an
embedder.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any

from app.processors.safe import process_text_safe
from app.schemas.ingest_analysis import (
    AnalyzedChunk,
    AnalyzedPrimitive,
    CandidateEvidence,
    DatasetCreationCandidate,
    DatasetRouteCandidate,
    ExistingDatasetSummary,
    GraphEdgeProposal,
    IngestAnalysisConfig,
    IngestAnalysisResult,
    PrimitiveWriteProposal,
    SessionMemoryWriteProposal,
)
from app.schemas.processor import ProcessorRequest
from app.store.search import cosine_similarity, tokenize

Embedder = Callable[[list[str]], list[list[float]]]

_BUILT_IN_RULES = (
    ("task", re.compile(r"\b(todo|fixme|bug|follow[- ]?up|refactor|test needed|migrate|migration|we need to|need to)\b", re.IGNORECASE), 0.68),
    ("decision", re.compile(r"\b(decided|decision|accepted|rejected|tradeoff|let'?s use|we will use|go with)\b", re.IGNORECASE), 0.7),
    ("constraint", re.compile(r"\b(must|must not|never|always|required|constraint|compatible|performance|security)\b", re.IGNORECASE), 0.66),
    ("preference", re.compile(r"\b(prefer|preference|rather|like|dislike|avoid|we want|i want)\b", re.IGNORECASE), 0.62),
    ("fact", re.compile(r"\b(is|are|supports|has|contains|uses|means)\b", re.IGNORECASE), 0.45),
    ("event", re.compile(r"\b(today|yesterday|tomorrow|last week|next week|meeting|call|happened|created|updated)\b", re.IGNORECASE), 0.55),
    ("error", re.compile(r"\b(error|exception|traceback|failed|failure|crash|timeout|stack trace)\b", re.IGNORECASE), 0.72),
    ("command", re.compile(r"(`[^`]+`|\b(python|pytest|uvicorn|curl|npm|git|pip)\s+[^\n.!?]+)", re.IGNORECASE), 0.72),
    ("reference", re.compile(r"(https?://\S+|\b[A-Za-z]:[\\/][^\s]+|\b[\w./-]+\.(py|ts|tsx|js|md|json|toml)\b)", re.IGNORECASE), 0.76),
    ("entity", re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*){0,3}\b"), 0.5),
)

_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "could",
    "from",
    "have",
    "into",
    "only",
    "should",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

_ENTITY_BLOCKLIST = {"i", "we", "todo", "this", "two", "last"}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_TEMPORAL_PATTERN = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}|"
    r"today|yesterday|tomorrow|"
    r"last\s+week|this\s+week|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago|"
    r"this\s+morning|this\s+afternoon|this\s+evening|tonight"
    r")\b",
    re.IGNORECASE,
)


def _digest(*parts: object, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:64] or "memory"


def _last_sunday(year: int, month: int) -> date:
    day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while day.weekday() != 6:
        day -= timedelta(days=1)
    return day


def _timezone(name: str, reference: datetime | None = None) -> tuple[tzinfo, str]:
    if name == "Europe/Berlin":
        ref_date = reference.date() if reference is not None else datetime.now().date()
        dst_start = _last_sunday(ref_date.year, 3)
        dst_end = _last_sunday(ref_date.year, 10)
        offset_hours = 2 if dst_start <= ref_date < dst_end else 1
        return timezone(timedelta(hours=offset_hours)), name
    if name.upper() == "UTC":
        return timezone.utc, "UTC"
    return timezone.utc, "UTC"


def _reference_now(config: IngestAnalysisConfig) -> datetime:
    if config.reference_now:
        raw = config.reference_now.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
        tz, _ = _timezone(config.timezone, parsed)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)
    tz, _ = _timezone(config.timezone)
    return datetime.now(tz)


def _day_window(day: date, tz: tzinfo) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _resolve_temporal(text: str, now: datetime, tz: tzinfo) -> tuple[datetime, datetime] | None:
    lowered = " ".join(text.lower().split())
    today = now.date()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", lowered):
        try:
            return _day_window(date.fromisoformat(lowered), tz)
        except ValueError:
            return None
    if lowered == "today":
        return _day_window(today, tz)
    if lowered == "yesterday":
        return _day_window(today - timedelta(days=1), tz)
    if lowered == "tomorrow":
        return _day_window(today + timedelta(days=1), tz)
    if lowered == "this morning":
        return (
            datetime.combine(today, time(hour=6), tzinfo=tz),
            datetime.combine(today, time(hour=12), tzinfo=tz),
        )
    if lowered == "this afternoon":
        return (
            datetime.combine(today, time(hour=12), tzinfo=tz),
            datetime.combine(today, time(hour=18), tzinfo=tz),
        )
    if lowered == "this evening":
        return (
            datetime.combine(today, time(hour=18), tzinfo=tz),
            datetime.combine(today, time(hour=22), tzinfo=tz),
        )
    if lowered == "tonight":
        return (
            datetime.combine(today, time(hour=18), tzinfo=tz),
            datetime.combine(today + timedelta(days=1), time(hour=6), tzinfo=tz),
        )
    if lowered == "this week":
        start_day = today - timedelta(days=today.weekday())
        start = datetime.combine(start_day, time.min, tzinfo=tz)
        return start, start + timedelta(days=7)
    if lowered == "last week":
        this_week_start = today - timedelta(days=today.weekday())
        start = datetime.combine(this_week_start - timedelta(days=7), time.min, tzinfo=tz)
        return start, start + timedelta(days=7)

    ago = re.fullmatch(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago", lowered)
    if ago:
        raw_days = ago.group(1)
        days = int(raw_days) if raw_days.isdigit() else _NUMBER_WORDS[raw_days]
        return _day_window(today - timedelta(days=days), tz)

    return None


def _extract_temporal_primitives(
    text: str,
    config: IngestAnalysisConfig,
) -> list[AnalyzedPrimitive]:
    now = _reference_now(config)
    tz, tz_label = _timezone(config.timezone, now)
    primitives: list[AnalyzedPrimitive] = []
    for match in _TEMPORAL_PATTERN.finditer(text):
        raw = match.group(0).strip()
        resolved = _resolve_temporal(raw, now, tz)
        if resolved is None:
            continue
        start, end = resolved
        primitives.append(
            AnalyzedPrimitive(
                id=f"prim-{_digest('time', match.start(), match.end(), raw)}",
                kind="time",
                text=raw,
                char_start=match.start(),
                char_end=match.end(),
                confidence=0.86,
                source="temporal_parser",
                metadata={
                    "resolved_start": _iso(start),
                    "resolved_end": _iso(end),
                    "timezone": tz_label,
                    "resolution_source": "logic_temporal_parser",
                },
            )
        )
    return primitives


def _dataset_text(dataset: ExistingDatasetSummary) -> str:
    return " ".join(
        part
        for part in (
            dataset.dataset_key,
            dataset.display_name,
            dataset.semantic_description,
            dataset.usage_guidance,
            dataset.llm_summary or "",
            " ".join(dataset.capability_tags),
            " ".join(dataset.entity_types),
        )
        if part
    )


def _keyword_overlap_score(query: str, target: str) -> float:
    query_tokens = {tok for tok in tokenize(query) if tok not in _STOPWORDS and len(tok) > 2}
    target_tokens = {tok for tok in tokenize(target) if tok not in _STOPWORDS and len(tok) > 2}
    if not query_tokens or not target_tokens:
        return 0.0
    overlap = query_tokens & target_tokens
    return round(len(overlap) / max(len(query_tokens), 1), 6)


def _chunk_source_ids(primitives: list[AnalyzedPrimitive], chunks: list[AnalyzedChunk]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for primitive in primitives:
        ids = [
            chunk.id
            for chunk in chunks
            if primitive.char_start >= chunk.char_start and primitive.char_end <= chunk.char_end
        ]
        mapping[primitive.id] = ids
    return mapping


def _extract_builtin_primitives(text: str, chunks: list[AnalyzedChunk]) -> list[AnalyzedPrimitive]:
    primitives: list[AnalyzedPrimitive] = []
    seen: set[tuple[str, int, int, str]] = set()
    for chunk in chunks:
        for kind, pattern, confidence in _BUILT_IN_RULES:
            for match in pattern.finditer(chunk.text):
                start = chunk.char_start + match.start()
                end = chunk.char_start + match.end()
                raw = text[start:end].strip()
                if not raw:
                    continue
                if kind == "entity" and (len(raw) <= 2 or raw.lower() in _ENTITY_BLOCKLIST):
                    continue
                key = (kind, start, end, raw.lower())
                if key in seen:
                    continue
                seen.add(key)
                primitives.append(
                    AnalyzedPrimitive(
                        id=f"prim-{_digest(kind, start, end, raw)}",
                        kind=kind,
                        text=raw,
                        char_start=start,
                        char_end=end,
                        confidence=confidence,
                        source="built_in",
                    )
                )
    return primitives


def _extract_custom_primitives(
    text: str,
    config: IngestAnalysisConfig,
) -> list[AnalyzedPrimitive]:
    primitives: list[AnalyzedPrimitive] = []
    for rule in config.custom_primitives:
        try:
            pattern = re.compile(rule.pattern, re.IGNORECASE)
        except re.error:
            continue
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            if not raw:
                continue
            primitives.append(
                AnalyzedPrimitive(
                    id=f"prim-{_digest(rule.kind, match.start(), match.end(), raw)}",
                    kind=rule.kind,
                    subkind=rule.subkind,
                    text=raw,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=rule.confidence,
                    source="custom",
                    target_dataset_key=rule.target_dataset_key,
                    metadata=dict(rule.metadata),
                )
            )
    return primitives


def _candidate_label(primitives: list[AnalyzedPrimitive], text: str) -> str:
    entity = next((p.text for p in primitives if p.kind == "entity" and len(p.text) > 2), None)
    if entity:
        return entity
    tokens = [tok for tok in tokenize(text) if tok not in _STOPWORDS and len(tok) > 3]
    if tokens:
        return " ".join(tokens[:3])
    return "session memory"


def _route_candidates(
    *,
    route_text: str,
    primitives: list[AnalyzedPrimitive],
    chunks: list[AnalyzedChunk],
    existing_datasets: list[ExistingDatasetSummary],
    config: IngestAnalysisConfig,
    embedder: Embedder | None,
) -> list[DatasetRouteCandidate]:
    explicit_keys = {p.target_dataset_key for p in primitives if p.target_dataset_key}
    vectors: list[list[float]] = []
    if embedder is not None and existing_datasets:
        vectors = embedder([route_text, *[_dataset_text(ds) for ds in existing_datasets]])

    query_vector = vectors[0] if vectors else None
    dataset_vectors = vectors[1:] if vectors else []
    routes: list[DatasetRouteCandidate] = []
    primitive_ids = [p.id for p in primitives]
    chunk_ids = [c.id for c in chunks]

    for index, dataset in enumerate(existing_datasets):
        target_text = _dataset_text(dataset)
        keyword_score = _keyword_overlap_score(route_text, target_text)
        vector_score = None
        if query_vector is not None and index < len(dataset_vectors):
            vector_score = round(max(0.0, cosine_similarity(query_vector, dataset_vectors[index])), 6)
        score = keyword_score
        if vector_score is not None:
            score = round((1.0 - config.vector_weight) * keyword_score + config.vector_weight * vector_score, 6)

        reasons: list[str] = []
        if keyword_score > 0:
            reasons.append(f"keyword_overlap:{keyword_score:.3f}")
        if vector_score is not None:
            reasons.append(f"vector_similarity:{vector_score:.3f}")
        if dataset.dataset_key in explicit_keys:
            score = max(score, config.route_threshold)
            reasons.append("custom_rule_target")

        if score >= config.route_threshold:
            routes.append(
                DatasetRouteCandidate(
                    dataset_key=dataset.dataset_key,
                    score=score,
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                    reasons=reasons,
                    primitive_ids=primitive_ids,
                    chunk_ids=chunk_ids,
                )
            )

    routes.sort(key=lambda route: route.score, reverse=True)
    return routes


def analyze_ingest(
    text: str,
    *,
    session_id: str = "main",
    config: IngestAnalysisConfig | dict[str, Any] | None = None,
    existing_datasets: list[ExistingDatasetSummary | dict[str, Any]] | None = None,
    candidate_state: list[CandidateEvidence | dict[str, Any]] | None = None,
    embedder: Embedder | None = None,
) -> IngestAnalysisResult:
    """Analyze text and return ingest proposals without mutating storage."""
    resolved_config = (
        config
        if isinstance(config, IngestAnalysisConfig)
        else IngestAnalysisConfig.model_validate(config or {})
    )
    datasets = [
        ds if isinstance(ds, ExistingDatasetSummary) else ExistingDatasetSummary.model_validate(ds)
        for ds in (existing_datasets or [])
    ]
    evidence_state = [
        ev if isinstance(ev, CandidateEvidence) else CandidateEvidence.model_validate(ev)
        for ev in (candidate_state or [])
    ]

    response = process_text_safe(
        ProcessorRequest(
            text=text,
            strategy="safe",
            max_chars=resolved_config.max_chars,
            overlap_chars=resolved_config.overlap_chars,
            extract_primitives=False,
        )
    )
    chunks = [
        AnalyzedChunk(
            id=f"chunk-{_digest(session_id, span.char_start, span.char_end, span.text)}",
            text=span.text,
            char_start=span.char_start,
            char_end=span.char_end,
            chunk_index=index,
            token_count=len(tokenize(span.text)),
            metadata={**span.metadata, "session_id": session_id},
        )
        for index, span in enumerate(response.chunks)
    ]

    primitives: list[AnalyzedPrimitive] = []
    if resolved_config.built_in_primitives_enabled:
        primitives.extend(_extract_builtin_primitives(text, chunks))
    if resolved_config.temporal_primitives_enabled:
        primitives.extend(_extract_temporal_primitives(text, resolved_config))
    primitives.extend(_extract_custom_primitives(text, resolved_config))
    primitives.sort(key=lambda item: (item.char_start, item.char_end, item.kind))

    raw_text_id = f"raw-{_digest(session_id, text)}"
    session_message_id = f"msg-{_digest(session_id, text)}"

    session_memory_writes = [
        SessionMemoryWriteProposal(
            dataset_key=resolved_config.session_memory_dataset_key,
            item_id=f"session-{_digest(session_id, chunk.id)}",
            raw_text=chunk.text,
            metadata={
                "session_id": session_id,
                "source": "logic_ingest_analysis",
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        )
        for chunk in chunks
    ]

    primitive_to_chunks = _chunk_source_ids(primitives, chunks)
    primitive_write_proposals = [
        PrimitiveWriteProposal(
            item_id=f"primitive-{_digest(session_id, primitive.id)}",
            kind=primitive.kind,
            raw_text=primitive.text,
            confidence=primitive.confidence,
            metadata={
                **primitive.metadata,
                "session_id": session_id,
                "raw_text_id": raw_text_id,
                "session_message_id": session_message_id,
                "primitive_id": primitive.id,
                "primitive_kind": primitive.kind,
                "primitive_subkind": primitive.subkind,
                "primitive_source": primitive.source,
                "target_dataset_key": primitive.target_dataset_key,
                "chunk_ids": primitive_to_chunks.get(primitive.id, []),
                "char_start": primitive.char_start,
                "char_end": primitive.char_end,
            },
        )
        for primitive in primitives
    ]

    route_text = " ".join([text, *[primitive.text for primitive in primitives]])
    routes = _route_candidates(
        route_text=route_text,
        primitives=primitives,
        chunks=chunks,
        existing_datasets=datasets,
        config=resolved_config,
        embedder=embedder,
    )

    label = _candidate_label(primitives, text)
    historical = {ev.label.lower(): ev.count for ev in evidence_state}
    current_evidence = max(1, len(primitives), len(chunks))
    evidence_count = current_evidence + historical.get(label.lower(), 0)
    ready = not routes and evidence_count >= resolved_config.min_candidate_evidence
    creation_candidates = [
        DatasetCreationCandidate(
            label=label,
            suggested_dataset_key=f"derived_{_slug(label)}" if ready else None,
            evidence_count=evidence_count,
            ready_to_create=ready,
            reasons=[
                "no_route_above_threshold" if not routes else "route_available",
                f"min_candidate_evidence:{resolved_config.min_candidate_evidence}",
            ],
            primitive_ids=[p.id for p in primitives],
            chunk_ids=[c.id for c in chunks],
        )
    ]

    graph_edges = [
        GraphEdgeProposal(
            source_type="raw_text",
            source_key=raw_text_id,
            target_type="session_message",
            target_key=session_message_id,
            edge_type="feeds_into",
            description="Raw text is represented by this session message proposal.",
        ),
        GraphEdgeProposal(
            source_type="session_message",
            source_key=session_message_id,
            target_type="session",
            target_key=session_id,
            edge_type="related",
            description="Session message belongs to this session.",
        ),
    ]
    graph_edges.extend(
        GraphEdgeProposal(
            source_type="session_message",
            source_key=session_message_id,
            target_type="memory_item",
            target_key=write.item_id,
            edge_type="produces",
            description="Session message produces this session-memory chunk proposal.",
        )
        for write in session_memory_writes
    )
    for primitive in primitives:
        primitive_item_id = f"primitive-{_digest(session_id, primitive.id)}"
        for chunk_id in primitive_to_chunks.get(primitive.id, []):
            graph_edges.append(
                GraphEdgeProposal(
                    source_type="memory_item",
                    source_key=chunk_id,
                    target_type="memory_item",
                    target_key=primitive_item_id,
                    edge_type="related",
                    description=f"Chunk contains extracted primitive kind={primitive.kind}.",
                )
            )
        if primitive.target_dataset_key:
            graph_edges.append(
                GraphEdgeProposal(
                    source_type="memory_item",
                    source_key=primitive_item_id,
                    target_type="dataset",
                    target_key=primitive.target_dataset_key,
                    edge_type="related",
                    description=f"Primitive has configured target dataset {primitive.target_dataset_key}.",
                )
            )
        if primitive.kind == "entity":
            graph_edges.append(
                GraphEdgeProposal(
                    source_type="memory_item",
                    source_key=primitive_item_id,
                    target_type="memory_item",
                    target_key=f"entity-{_slug(primitive.text)}",
                    edge_type="shared_entity",
                    description="Primitive proposes a reusable entity node.",
                )
            )
        if primitive.kind in {"fact", "framework", "cortexdb_concept"}:
            graph_edges.append(
                GraphEdgeProposal(
                    source_type="memory_item",
                    source_key=primitive_item_id,
                    target_type="memory_item",
                    target_key=f"knowledge-{_digest(primitive.kind, primitive.text)}",
                    edge_type="produces",
                    description="Primitive can be promoted into a knowledge item.",
                )
            )
    for route in routes:
        for primitive in primitives:
            if primitive.target_dataset_key != route.dataset_key:
                continue
            graph_edges.append(
                GraphEdgeProposal(
                    source_type="memory_item",
                    source_key=f"primitive-{_digest(session_id, primitive.id)}",
                    target_type="dataset",
                    target_key=route.dataset_key,
                    edge_type="related",
                    description=f"Primitive participates in route candidate {route.dataset_key}.",
                )
            )

    return IngestAnalysisResult(
        session_id=session_id,
        chunks=chunks,
        primitives=primitives,
        session_memory_writes=session_memory_writes,
        primitive_write_proposals=primitive_write_proposals,
        dataset_routes=routes,
        dataset_creation_candidates=creation_candidates,
        graph_edges=graph_edges,
        metadata={
            "proposal_only": True,
            "llm_used": False,
            "embedding_used": embedder is not None,
            "processor": response.processor,
            "processor_version": response.processor_version,
        },
    )
