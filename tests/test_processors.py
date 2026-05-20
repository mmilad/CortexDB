from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ingest.service import build_ingest_items_with_processor
from app.processors.api import app as processor_app
from app.processors.safe import process_text_safe
from app.processors.config import ProcessorConfig
from app.processors.validation import ProcessorValidationError, validate_processor_response
from app.schemas.processor import ProcessorEntity, ProcessorRequest, ProcessorResponse, ProcessorSpan


class DisabledProcessor:
    def is_enabled(self) -> bool:
        return False


class SafeProcessor:
    def is_enabled(self) -> bool:
        return True

    async def process_text(self, request: ProcessorRequest) -> ProcessorResponse:
        return process_text_safe(request)


def test_processor_validation_accepts_valid_offsets() -> None:
    source = "First sentence. Second sentence."
    response = ProcessorResponse(
        processor="test",
        processor_version="test/1",
        strategy="safe",
        chunks=[
            ProcessorSpan(
                text="First sentence.",
                char_start=0,
                char_end=15,
                primitive="chunk",
            )
        ],
    )

    assert validate_processor_response(source, response, max_chars=100) is response


def test_processor_validation_rejects_invalid_offsets() -> None:
    source = "First sentence."
    response = ProcessorResponse(
        processor="test",
        processor_version="test/1",
        strategy="safe",
        chunks=[
            ProcessorSpan(
                text="First sentence.",
                char_start=0,
                char_end=99,
                primitive="chunk",
            )
        ],
    )

    with pytest.raises(ProcessorValidationError, match="outside source"):
        validate_processor_response(source, response, max_chars=100)


def test_processor_validation_rejects_rewritten_text() -> None:
    source = "First sentence."
    response = ProcessorResponse(
        processor="test",
        processor_version="test/1",
        strategy="safe",
        chunks=[
            ProcessorSpan(
                text="Changed sentence.",
                char_start=0,
                char_end=15,
                primitive="chunk",
            )
        ],
    )

    with pytest.raises(ProcessorValidationError, match="does not match"):
        validate_processor_response(source, response, max_chars=100)


def test_processor_validation_rejects_empty_chunks() -> None:
    source = "First sentence."
    response = ProcessorResponse(
        processor="test",
        processor_version="test/1",
        strategy="safe",
        chunks=[],
    )

    with pytest.raises(ProcessorValidationError, match="no chunks"):
        validate_processor_response(source, response, max_chars=100)


def test_processor_validation_rejects_invalid_entity_offsets() -> None:
    source = "Mastra is useful."
    response = ProcessorResponse(
        processor="test",
        processor_version="test/1",
        strategy="semantic",
        chunks=[ProcessorSpan(text=source, char_start=0, char_end=len(source))],
        entities=[
            ProcessorEntity(
                text="Mastra!",
                label="PRODUCT",
                char_start=0,
                char_end=6,
            )
        ],
    )

    with pytest.raises(ProcessorValidationError, match="span text does not match"):
        validate_processor_response(source, response, max_chars=100)


def test_processor_config_prefers_client_env_names(monkeypatch) -> None:
    monkeypatch.setenv("CORTEXDB_PROCESSOR_PROVIDER", "local")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_URL", "http://legacy:5010")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_STRATEGY", "safe")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_CLIENT_PROVIDER", "sidecar")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_CLIENT_URL", "http://client:5010")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_CLIENT_STRATEGY", "semantic")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_CLIENT_CLASSIFY", "true")

    config = ProcessorConfig.from_env()

    assert config.provider == "sidecar"
    assert config.url == "http://client:5010"
    assert config.strategy == "semantic"
    assert config.classify is True


def test_processor_config_supports_legacy_env_names(monkeypatch) -> None:
    monkeypatch.delenv("CORTEXDB_PROCESSOR_CLIENT_PROVIDER", raising=False)
    monkeypatch.delenv("CORTEXDB_PROCESSOR_CLIENT_URL", raising=False)
    monkeypatch.delenv("CORTEXDB_PROCESSOR_CLIENT_STRATEGY", raising=False)
    monkeypatch.setenv("CORTEXDB_PROCESSOR_PROVIDER", "sidecar")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_URL", "http://legacy:5010")
    monkeypatch.setenv("CORTEXDB_PROCESSOR_STRATEGY", "safe")

    config = ProcessorConfig.from_env()

    assert config.provider == "sidecar"
    assert config.url == "http://legacy:5010"
    assert config.strategy == "safe"


@pytest.mark.asyncio
async def test_processor_disabled_uses_fallback_chunker() -> None:
    items, job = await build_ingest_items_with_processor(
        "Alpha paragraph.\n\nBeta paragraph.",
        DisabledProcessor(),  # type: ignore[arg-type]
        processor_strategy="safe",
        max_chars=100,
        overlap_chars=0,
    )

    assert len(items) == 1
    assert job is not None
    assert job.status == "skipped"
    assert items[0].metadata["source_type"] == "text"
    assert "char_start" not in items[0].metadata


@pytest.mark.asyncio
async def test_safe_processor_builds_offset_items_and_primitives() -> None:
    items, job = await build_ingest_items_with_processor(
        "TODO: write tests. We decided to keep the sidecar.",
        SafeProcessor(),  # type: ignore[arg-type]
        processor_strategy="safe",
        extract_primitives=True,
        max_chars=100,
        overlap_chars=0,
    )

    assert job is not None
    assert job.status == "completed"
    assert job.primitive_count >= 1
    assert items[0].metadata["char_start"] == 0
    assert items[0].raw_text == "TODO: write tests. We decided to keep the sidecar."
    primitive_items = [item for item in items if item.id and item.id.startswith("primitive-")]
    assert primitive_items
    assert primitive_items[0].metadata["source"] == "processor_extraction"


def test_processor_sidecar_health() -> None:
    client = TestClient(processor_app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_processor_sidecar_process_text_safe() -> None:
    client = TestClient(processor_app)
    response = client.post(
        "/process/text",
        json={
            "text": "TODO: write tests. Done.",
            "strategy": "safe",
            "max_chars": 100,
            "overlap_chars": 0,
            "extract_primitives": True,
            "metadata": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "safe"
    assert body["chunks"][0]["text"] == "TODO: write tests. Done."
    assert body["primitives"][0]["kind"] == "task"
