"""Source loading helpers for ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source_type: str
    source_identity: str
    source_path: str | None = None
    filename: str | None = None


def _read_file(path: Path) -> SourceDocument:
    resolved = path.resolve()
    return SourceDocument(
        text=resolved.read_text(encoding="utf-8"),
        source_type="file",
        source_identity=str(resolved),
        source_path=str(resolved),
        filename=resolved.name,
    )


def iter_source_documents(source: str | Path) -> list[SourceDocument]:
    """Load a text source, supported file, or directory of supported files."""
    if isinstance(source, str):
        return [
            SourceDocument(
                text=source,
                source_type="text",
                source_identity="text",
            )
        ]

    path = Path(source)
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        return [_read_file(p) for p in files]

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return []
        return [_read_file(path)]

    raise FileNotFoundError(f"source not found: {path}")
