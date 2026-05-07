from __future__ import annotations

import os

import pytest

from app.ingest import build_ingest_items, chunk_text, ingest_source_to_dataset
from app.store import SqliteStore

os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"


class FakeEmbeddingService:
    model_id = "fake/test-embedding"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def is_enabled(self) -> bool:
        return True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]


@pytest.fixture()
def store(tmp_path):
    s = SqliteStore(str(tmp_path / "test.sqlite"))
    yield s
    s.close()


def test_chunk_text_preserves_paragraphs_where_practical() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text(text, max_chars=35, overlap_chars=5)

    assert chunks[0] == "First paragraph.\n\nSecond paragraph."
    assert chunks[1].endswith("Third paragraph.")
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_applies_overlap_for_long_text() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", max_chars=10, overlap_chars=3)

    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith(chunks[0][-3:])
    assert all(0 < len(chunk) <= 10 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text("hello", max_chars=10, overlap_chars=10)


def test_build_ingest_items_metadata_and_stable_ids_for_text() -> None:
    text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."

    first = build_ingest_items(
        text,
        max_chars=25,
        overlap_chars=5,
        metadata={"source_type": "caller", "topic": "demo"},
    )
    second = build_ingest_items(
        text,
        max_chars=25,
        overlap_chars=5,
        metadata={"source_type": "caller", "topic": "demo"},
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert first[0].metadata["source_type"] == "text"
    assert first[0].metadata["topic"] == "demo"
    assert first[0].metadata["chunk_index"] == 0
    assert first[0].metadata["chunk_count"] == len(first)
    assert first[0].metadata["content_sha256"]
    assert first[0].metadata["source_sha256"]
    assert first[0].metadata["ingestion_id"].startswith("ingest-")


def test_build_ingest_items_hashes_change_when_content_changes() -> None:
    first = build_ingest_items("same source text", max_chars=100, overlap_chars=0)
    second = build_ingest_items("changed source text", max_chars=100, overlap_chars=0)

    assert first[0].metadata["source_sha256"] != second[0].metadata["source_sha256"]
    assert first[0].metadata["content_sha256"] != second[0].metadata["content_sha256"]
    assert first[0].id != second[0].id


def test_build_ingest_items_file_metadata(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")

    items = build_ingest_items(path, max_chars=100, overlap_chars=0)

    assert len(items) == 1
    metadata = items[0].metadata
    assert metadata["source_type"] == "file"
    assert metadata["source_path"] == str(path.resolve())
    assert metadata["filename"] == "notes.md"


def test_directory_build_ignores_unsupported_files_and_is_sorted(tmp_path) -> None:
    (tmp_path / "b.md").write_text("B doc", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A doc", encoding="utf-8")
    (tmp_path / "ignored.json").write_text('{"x": 1}', encoding="utf-8")

    items = build_ingest_items(tmp_path, max_chars=100, overlap_chars=0)

    assert [item.raw_text for item in items] == ["A doc", "B doc"]
    assert [item.metadata["filename"] for item in items] == ["a.txt", "b.md"]


@pytest.mark.asyncio
async def test_ingest_source_to_dataset_batches_and_stores_items(store: SqliteStore) -> None:
    store.upsert_dataset("docs", {"dataset_key": "docs", "display_name": "Docs"})
    embed_svc = FakeEmbeddingService()

    result = await ingest_source_to_dataset(
        "docs",
        "abcdefghijklmnopqrstuvwxyz" * 4,
        store,
        embed_svc,  # type: ignore[arg-type]
        max_chars=25,
        overlap_chars=5,
        batch_size=2,
    )

    assert result.ingested > 2
    assert embed_svc.batch_sizes[0] == 2
    assert store.count_memory_items("docs") == result.ingested
    stored = store.list_memory_items("docs", limit=10)
    assert stored[0]["embedding_model"] == "fake/test-embedding"
