"""Deterministic paragraph-aware text chunking."""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def _validate_chunk_config(max_chars: int, overlap_chars: int) -> None:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars")


def _join_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n{right}"


def _tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    return text[-overlap_chars:].strip()


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    step = max_chars - overlap_chars
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_text(text: str, *, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Split text into non-empty chunks, preserving paragraphs where practical."""
    _validate_chunk_config(max_chars, overlap_chars)

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(normalized) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(paragraph, max_chars, overlap_chars))
            continue

        candidate = _join_parts(current, paragraph)
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        prefix = _tail(current, overlap_chars)
        candidate = _join_parts(prefix, paragraph)
        current = candidate if len(candidate) <= max_chars else paragraph

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]
