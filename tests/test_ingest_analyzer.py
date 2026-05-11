from __future__ import annotations

from fastapi.testclient import TestClient

from app.ingest.analyzer import analyze_ingest
from app.processors.api import app as processor_app
from app.schemas.ingest_analysis import IngestAnalysisConfig


def test_analyze_ingest_proposes_session_memory_chunks() -> None:
    result = analyze_ingest(
        "TODO: write tests. We decided to keep the sidecar.",
        session_id="planning",
        config={"max_chars": 100, "overlap_chars": 0},
    )

    assert result.session_id == "planning"
    assert result.metadata["proposal_only"] is True
    assert result.metadata["llm_used"] is False
    assert result.chunks
    assert result.session_memory_writes
    assert result.session_memory_writes[0].dataset_key == "session_memory"
    assert result.session_memory_writes[0].metadata["session_id"] == "planning"
    assert result.graph_edges


def test_analyze_ingest_keeps_stable_offsets_and_spans() -> None:
    text = "First sentence. Second sentence."
    result = analyze_ingest(text, config={"max_chars": 100, "overlap_chars": 0})

    assert result.chunks[0].text == text
    assert result.chunks[0].char_start == 0
    assert result.chunks[0].char_end == len(text)
    assert text[result.chunks[0].char_start : result.chunks[0].char_end] == result.chunks[0].text


def test_analyze_ingest_extracts_builtin_primitives() -> None:
    result = analyze_ingest(
        "TODO: migrate the database. We must keep compatibility. I prefer Mastra.",
        config={"max_chars": 100, "overlap_chars": 0},
    )

    kinds = {primitive.kind for primitive in result.primitives}
    assert "task" in kinds
    assert "constraint" in kinds
    assert "Mastra" in {primitive.text for primitive in result.primitives}
    assert "We" not in {primitive.text for primitive in result.primitives}
    assert "I" not in {primitive.text for primitive in result.primitives}


def test_analyze_ingest_resolves_temporal_primitives() -> None:
    result = analyze_ingest(
        "Yesterday we discussed Mastra. Two days ago we compared LangChain. "
        "Last week we made notes. This morning we checked 2026-05-01.",
        config={
            "max_chars": 300,
            "overlap_chars": 0,
            "reference_now": "2026-05-08T19:00:00+02:00",
            "timezone": "Europe/Berlin",
        },
    )

    times = {primitive.text.lower(): primitive for primitive in result.primitives if primitive.kind == "time"}
    assert times["yesterday"].metadata["resolved_start"] == "2026-05-07T00:00:00+02:00"
    assert times["two days ago"].metadata["resolved_start"] == "2026-05-06T00:00:00+02:00"
    assert times["last week"].metadata["resolved_start"] == "2026-04-27T00:00:00+02:00"
    assert times["this morning"].metadata["resolved_start"] == "2026-05-08T06:00:00+02:00"
    assert times["2026-05-01"].metadata["resolved_end"] == "2026-05-02T00:00:00+02:00"
    assert times["yesterday"].metadata["resolution_source"] == "logic_temporal_parser"


def test_analyze_ingest_creates_primitive_write_and_graph_proposals() -> None:
    result = analyze_ingest(
        "Yesterday Mastra supported RAG.",
        config={
            "max_chars": 100,
            "overlap_chars": 0,
            "reference_now": "2026-05-08T19:00:00+02:00",
            "custom_primitives": [
                {"kind": "framework", "pattern": r"\bMastra\b", "target_dataset_key": "frameworks"}
            ],
        },
        existing_datasets=[
            {
                "dataset_key": "frameworks",
                "display_name": "Frameworks",
                "semantic_description": "Mastra RAG framework notes.",
            }
        ],
    )

    assert result.primitive_write_proposals
    time_write = next(write for write in result.primitive_write_proposals if write.kind == "time")
    assert time_write.metadata["resolved_start"] == "2026-05-07T00:00:00+02:00"
    edge_shapes = {(edge.source_type, edge.target_type, edge.edge_type) for edge in result.graph_edges}
    assert ("raw_text", "session_message", "feeds_into") in edge_shapes
    assert ("session_message", "memory_item", "produces") in edge_shapes
    assert any(edge.target_type == "dataset" and edge.target_key == "frameworks" for edge in result.graph_edges)


def test_analyze_ingest_applies_custom_regex_primitives() -> None:
    result = analyze_ingest(
        "We discussed Mastra and LangChain for built-in RAG.",
        config={
            "max_chars": 100,
            "overlap_chars": 0,
            "custom_primitives": [
                {
                    "kind": "framework",
                    "pattern": r"\b(Mastra|LangChain)\b",
                    "target_dataset_key": "frameworks",
                }
            ],
        },
        existing_datasets=[
            {
                "dataset_key": "frameworks",
                "display_name": "Frameworks",
                "semantic_description": "Agent frameworks and RAG libraries.",
            }
        ],
    )

    framework_primitives = [p for p in result.primitives if p.kind == "framework"]
    assert {p.text for p in framework_primitives} == {"Mastra", "LangChain"}
    assert result.dataset_routes[0].dataset_key == "frameworks"
    assert "custom_rule_target" in result.dataset_routes[0].reasons


def test_analyze_ingest_routes_to_existing_dataset_by_keyword() -> None:
    result = analyze_ingest(
        "Mastra RAG framework",
        existing_datasets=[
            {
                "dataset_key": "framework_knowledge",
                "display_name": "Framework Knowledge",
                "semantic_description": "Mastra RAG framework notes and comparisons.",
            }
        ],
        config={"route_threshold": 0.35},
    )

    assert result.dataset_routes
    assert result.dataset_routes[0].dataset_key == "framework_knowledge"
    assert result.dataset_routes[0].keyword_score >= 0.35
    assert result.dataset_creation_candidates[0].ready_to_create is False


def test_analyze_ingest_routes_with_fake_embedder() -> None:
    def fake_embedder(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "vector target" in text.lower() or text == texts[0]:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    result = analyze_ingest(
        "unmatched semantic phrase",
        existing_datasets=[
            {
                "dataset_key": "semantic_match",
                "display_name": "Vector Target",
                "semantic_description": "No shared keywords here.",
            }
        ],
        config={"route_threshold": 0.5, "vector_weight": 0.8},
        embedder=fake_embedder,
    )

    assert result.metadata["embedding_used"] is True
    assert result.dataset_routes[0].dataset_key == "semantic_match"
    assert result.dataset_routes[0].vector_score == 1.0


def test_analyze_ingest_marks_dataset_candidate_ready_after_repeated_evidence() -> None:
    result = analyze_ingest(
        "QuantumFlux handles blue vector timing.",
        config={"min_candidate_evidence": 3},
        candidate_state=[{"label": "QuantumFlux", "count": 2}],
    )

    candidate = result.dataset_creation_candidates[0]
    assert candidate.label == "QuantumFlux"
    assert candidate.ready_to_create is True
    assert candidate.suggested_dataset_key == "derived_quantumflux"


def test_analyze_ingest_does_not_require_database_writes(monkeypatch) -> None:
    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("analyzer must not touch SqliteStore")

    monkeypatch.setattr("app.store.main.SqliteStore.insert_memory_item", fail)
    result = analyze_ingest("TODO: keep this proposal-only.")

    assert result.session_memory_writes
    assert result.primitive_write_proposals
    assert result.metadata["proposal_only"] is True


def test_processor_sidecar_analyze_ingest_endpoint_matches_python_function() -> None:
    payload = {
        "text": "TODO: compare Mastra RAG.",
        "session_id": "main",
        "config": {"max_chars": 100, "overlap_chars": 0},
        "existing_datasets": [
            {
                "dataset_key": "frameworks",
                "display_name": "Frameworks",
                "semantic_description": "Mastra RAG and LangChain notes.",
            }
        ],
    }
    client = TestClient(processor_app)
    response = client.post("/analyze/ingest", json=payload)
    direct = analyze_ingest(
        payload["text"],
        session_id=payload["session_id"],
        config=IngestAnalysisConfig.model_validate(payload["config"]),
        existing_datasets=payload["existing_datasets"],
    )

    assert response.status_code == 200
    assert response.json() == direct.model_dump(mode="json")
