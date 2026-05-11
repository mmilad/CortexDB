"""Terminal playground for the logic-only ingest analyzer.

Edit ingest_config.py, then run:
    python run_ingest.py
"""

from __future__ import annotations

import textwrap

from app.ingest import analyze_ingest
from ingest_config import CANDIDATE_STATE, CONFIG, EXISTING_DATASETS, SAMPLE_TEXT, SESSION_ID


def _line(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _indent(text: str, width: int = 84) -> str:
    return textwrap.fill(" ".join(text.split()), width=width, subsequent_indent="    ")


def _print_config() -> None:
    _line("CONFIG")
    print(f"session_id: {SESSION_ID}")
    print(f"session_memory_dataset_key: {CONFIG.session_memory_dataset_key}")
    print(f"max_chars / overlap_chars: {CONFIG.max_chars} / {CONFIG.overlap_chars}")
    print(f"timezone: {CONFIG.timezone}")
    print(f"reference_now: {CONFIG.reference_now}")
    print(f"temporal_primitives_enabled: {CONFIG.temporal_primitives_enabled}")
    print(f"route_threshold: {CONFIG.route_threshold}")
    print(f"min_candidate_evidence: {CONFIG.min_candidate_evidence}")
    print("custom primitives:")
    for rule in CONFIG.custom_primitives:
        target = f" -> {rule.target_dataset_key}" if rule.target_dataset_key else ""
        print(f"  - {rule.kind}{target}: {rule.pattern}")


def _print_input() -> None:
    _line("INPUT TEXT")
    print(SAMPLE_TEXT.strip())


def _print_chunks(result) -> None:  # noqa: ANN001
    _line("CHUNKS")
    for chunk in result.chunks:
        print(f"[{chunk.chunk_index}] {chunk.id} chars={chunk.char_start}:{chunk.char_end} tokens={chunk.token_count}")
        print(f"    {_indent(chunk.text)}")


def _print_primitives(result) -> None:  # noqa: ANN001
    _line("PRIMITIVES")
    if not result.primitives:
        print("(none)")
        return
    for primitive in result.primitives:
        target = f" target={primitive.target_dataset_key}" if primitive.target_dataset_key else ""
        subkind = f" subkind={primitive.subkind}" if primitive.subkind else ""
        print(
            f"- {primitive.kind}{subkind} source={primitive.source} "
            f"confidence={primitive.confidence:.2f}{target}"
        )
        print(f"  span={primitive.char_start}:{primitive.char_end} id={primitive.id}")
        print(f"  text: {primitive.text}")
        if primitive.kind == "time":
            print(
                "  resolved: "
                f"{primitive.metadata.get('resolved_start')} -> {primitive.metadata.get('resolved_end')} "
                f"({primitive.metadata.get('timezone')})"
            )


def _print_session_writes(result) -> None:  # noqa: ANN001
    _line("SESSION MEMORY WRITE PROPOSALS")
    for write in result.session_memory_writes:
        print(f"- dataset={write.dataset_key} item_id={write.item_id}")
        print(f"  metadata={write.metadata}")


def _print_primitive_writes(result) -> None:  # noqa: ANN001
    _line("PRIMITIVE WRITE PROPOSALS")
    if not result.primitive_write_proposals:
        print("(none)")
        return
    for write in result.primitive_write_proposals:
        print(f"- kind={write.kind} item_id={write.item_id} confidence={write.confidence:.2f}")
        print(f"  raw_text={write.raw_text}")
        print(f"  metadata={write.metadata}")


def _print_routes(result) -> None:  # noqa: ANN001
    _line("DATASET ROUTES")
    if not result.dataset_routes:
        print("(none above threshold)")
        return
    for route in result.dataset_routes:
        vector = "none" if route.vector_score is None else f"{route.vector_score:.3f}"
        print(
            f"- {route.dataset_key}: score={route.score:.3f} "
            f"keyword={route.keyword_score:.3f} vector={vector}"
        )
        print(f"  reasons={route.reasons}")


def _print_candidates(result) -> None:  # noqa: ANN001
    _line("DATASET CREATION CANDIDATES")
    for candidate in result.dataset_creation_candidates:
        print(
            f"- label={candidate.label!r} evidence={candidate.evidence_count} "
            f"ready_to_create={candidate.ready_to_create}"
        )
        print(f"  suggested_dataset_key={candidate.suggested_dataset_key}")
        print(f"  reasons={candidate.reasons}")


def _print_graph_edges(result) -> None:  # noqa: ANN001
    _line("GRAPH EDGE PROPOSALS")
    for edge in result.graph_edges:
        print(
            f"- {edge.source_type}:{edge.source_key} "
            f"-[{edge.edge_type}]-> {edge.target_type}:{edge.target_key}"
        )
        print(f"  {edge.description}")


def main() -> None:
    result = analyze_ingest(
        SAMPLE_TEXT,
        session_id=SESSION_ID,
        config=CONFIG,
        existing_datasets=EXISTING_DATASETS,
        candidate_state=CANDIDATE_STATE,
    )

    _print_config()
    _print_input()
    _print_chunks(result)
    _print_primitives(result)
    _print_session_writes(result)
    _print_primitive_writes(result)
    _print_routes(result)
    _print_candidates(result)
    _print_graph_edges(result)

    _line("METADATA")
    print(result.metadata)


if __name__ == "__main__":
    main()
