"""Editable playground config for the logic-only ingest analyzer.

Run with:
    python run_ingest.py

Change SAMPLE_TEXT, CUSTOM_PRIMITIVES, EXISTING_DATASETS, or CANDIDATE_STATE
to see how the proposal changes. This file is intentionally plain Python so
you can tweak rules quickly while the prototype is still experimental.
"""

from __future__ import annotations

from app.schemas.ingest_analysis import (
    CandidateEvidence,
    CustomPrimitiveRule,
    ExistingDatasetSummary,
    IngestAnalysisConfig,
)

SESSION_ID = "main"

SAMPLE_TEXT = """\
This morning we talked about using Mastra or LangChain as inspiration for CortexDB.
I want CortexDB ingest to stay logic-only and avoid LLM calls during /ingest.
TODO: create a separate run_ingest.py playground and make framework mentions route to the frameworks dataset.
Two days ago we decided that repeated unmatched topics should become dataset candidates, not real datasets immediately.
Last week we discussed GraphRAG, and 2026-05-01 should resolve as an explicit date.
"""

CONFIG = IngestAnalysisConfig(
    reference_now="2026-05-08T19:00:00+02:00",
    timezone="Europe/Berlin",
    max_chars=280,
    overlap_chars=0,
    session_memory_dataset_key="session_memory",
    route_threshold=0.25,
    vector_weight=0.35,
    min_candidate_evidence=3,
    custom_primitives=[
        CustomPrimitiveRule(
            kind="framework",
            pattern=r"\b(Mastra|LangChain|LlamaIndex|Haystack)\b",
            target_dataset_key="frameworks",
            confidence=0.82,
            metadata={"domain": "agent_frameworks"},
        ),
        CustomPrimitiveRule(
            kind="cortexdb_concept",
            pattern=r"\b(GraphRAG|RAG|ingest|dataset candidates?|session memory|logic-only)\b",
            target_dataset_key="cortexdb_design",
            confidence=0.74,
            metadata={"domain": "cortexdb"},
        ),
    ],
)

EXISTING_DATASETS = [
    ExistingDatasetSummary(
        dataset_key="frameworks",
        display_name="Frameworks",
        semantic_description="Agent frameworks, RAG libraries, and comparisons such as Mastra, LangChain, LlamaIndex, and Haystack.",
        usage_guidance="Use for notes about external frameworks and what CortexDB can learn from them.",
        capability_tags=["frameworks", "rag", "agents"],
        entity_types=["Framework", "Library"],
        retrieval_capabilities=["keyword", "vector"],
    ),
    ExistingDatasetSummary(
        dataset_key="cortexdb_design",
        display_name="CortexDB Design",
        semantic_description="Architecture decisions, ingest behavior, retrieval design, GraphRAG middleware, and memory pipeline notes.",
        usage_guidance="Use for design discussions about how CortexDB should store, route, and retrieve memory.",
        capability_tags=["architecture", "ingest", "retrieval"],
        entity_types=["Decision", "Constraint", "DesignNote"],
        retrieval_capabilities=["keyword", "vector"],
    ),
    ExistingDatasetSummary(
        dataset_key="tasks",
        display_name="Tasks",
        semantic_description="Action items, TODOs, bugs, follow-ups, migrations, and implementation work.",
        usage_guidance="Use when an ingest message contains concrete work to do.",
        capability_tags=["tasks", "todo", "work"],
        entity_types=["Task"],
        retrieval_capabilities=["keyword", "vector"],
    ),
]

# Simulates previously observed unmatched evidence. Increase counts to see
# dataset candidates become ready_to_create.
CANDIDATE_STATE = [
    CandidateEvidence(label="Mastra", count=1),
    CandidateEvidence(label="logic only ingest", count=2),
]
