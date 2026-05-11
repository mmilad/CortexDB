"""Safe deterministic processor implementation used by the sidecar."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass

from app.schemas.processor import ProcessorPrimitive, ProcessorRequest, ProcessorResponse, ProcessorSpan

_SENTENCE_FALLBACK_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_TASK_RE = re.compile(r"\b(todo|fixme|bug|follow[- ]?up|refactor|test needed|migrate|migration)\b", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(decided|decision|accepted|rejected|tradeoff)\b", re.IGNORECASE)
_CONSTRAINT_RE = re.compile(r"\b(must|must not|never|always|required|constraint|compatible|performance|security)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


def _processor_version() -> str:
    try:
        return f"spacy/{importlib.metadata.version('spacy')}"
    except importlib.metadata.PackageNotFoundError:
        return "fallback-regex/1"


def _spacy_sentence_spans(text: str) -> list[_Span] | None:
    try:
        import spacy
    except ImportError:
        return None

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return [_Span(sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]


def _regex_sentence_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _SENTENCE_FALLBACK_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append(_Span(start, end))
    return spans


def _sentence_spans(text: str) -> list[_Span]:
    spans = _spacy_sentence_spans(text)
    return spans if spans is not None else _regex_sentence_spans(text)


def _hard_split_span(span: _Span, *, max_chars: int, overlap_chars: int) -> list[_Span]:
    chunks: list[_Span] = []
    step = max_chars - overlap_chars
    start = span.start
    while start < span.end:
        end = min(start + max_chars, span.end)
        chunks.append(_Span(start, end))
        if end >= span.end:
            break
        start += step
    return chunks


def _trim_span(text: str, span: _Span) -> _Span | None:
    start, end = span.start, span.end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return _Span(start, end)


def _pack_sentence_chunks(text: str, spans: list[_Span], *, max_chars: int, overlap_chars: int) -> list[ProcessorSpan]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars")

    packed: list[_Span] = []
    current: _Span | None = None
    for sentence in spans:
        if sentence.end - sentence.start > max_chars:
            if current is not None:
                packed.append(current)
                current = None
            packed.extend(_hard_split_span(sentence, max_chars=max_chars, overlap_chars=overlap_chars))
            continue

        if current is None:
            current = sentence
            continue

        candidate = _Span(current.start, sentence.end)
        if candidate.end - candidate.start <= max_chars:
            current = candidate
            continue

        packed.append(current)
        if overlap_chars > 0:
            overlap_start = max(current.start, current.end - overlap_chars)
            current = _Span(overlap_start, sentence.end)
            if current.end - current.start > max_chars:
                current = sentence
        else:
            current = sentence

    if current is not None:
        packed.append(current)

    chunks: list[ProcessorSpan] = []
    for span in packed:
        trimmed = _trim_span(text, span)
        if trimmed is None:
            continue
        chunks.append(
            ProcessorSpan(
                text=text[trimmed.start:trimmed.end],
                char_start=trimmed.start,
                char_end=trimmed.end,
                primitive="chunk",
                metadata={"boundary": "sentence"},
            )
        )
    return chunks


def _subkind(kind: str, text: str) -> str | None:
    lowered = text.lower()
    if kind == "task":
        if "bug" in lowered or "fixme" in lowered:
            return "bug"
        if "follow" in lowered:
            return "follow_up"
        if "migration" in lowered or "migrate" in lowered:
            return "migration"
        if "refactor" in lowered:
            return "refactor"
        if "test needed" in lowered:
            return "test_needed"
        return "todo"
    if kind == "decision":
        if "accepted" in lowered:
            return "accepted"
        if "rejected" in lowered:
            return "rejected"
        if "tradeoff" in lowered:
            return "tradeoff"
        return "pending"
    if kind == "constraint":
        if "security" in lowered:
            return "security"
        if "performance" in lowered:
            return "performance"
        if "compatible" in lowered:
            return "compatibility"
        return "architecture"
    return None


def _extract_rule_primitives(text: str, spans: list[_Span]) -> list[ProcessorPrimitive]:
    primitives: list[ProcessorPrimitive] = []
    rules = (("task", _TASK_RE), ("decision", _DECISION_RE), ("constraint", _CONSTRAINT_RE))
    for span in spans:
        raw = text[span.start:span.end]
        for kind, pattern in rules:
            if not pattern.search(raw):
                continue
            primitives.append(
                ProcessorPrimitive(
                    kind=kind,  # type: ignore[arg-type]
                    subkind=_subkind(kind, raw),
                    text=raw,
                    char_start=span.start,
                    char_end=span.end,
                    confidence=0.65,
                    metadata={"extractor": "rules"},
                )
            )
            break
    return primitives


def process_text_safe(request: ProcessorRequest) -> ProcessorResponse:
    spans = _sentence_spans(request.text)
    if not spans:
        trimmed = _trim_span(request.text, _Span(0, len(request.text)))
        spans = [trimmed] if trimmed else []

    chunks = _pack_sentence_chunks(
        request.text,
        spans,
        max_chars=request.max_chars,
        overlap_chars=request.overlap_chars,
    )
    primitives = _extract_rule_primitives(request.text, spans) if request.extract_primitives else []
    return ProcessorResponse(
        processor="spacy" if _processor_version().startswith("spacy/") else "fallback-regex",
        processor_version=_processor_version(),
        strategy=request.strategy,
        chunks=chunks,
        primitives=primitives,
    )
