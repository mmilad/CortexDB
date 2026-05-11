from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomPrimitiveRule(BaseModel):
    kind: str = Field(..., min_length=1)
    pattern: str = Field(..., min_length=1)
    subkind: str | None = None
    target_dataset_key: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateEvidence(BaseModel):
    label: str
    count: int = Field(default=0, ge=0)


class IngestAnalysisConfig(BaseModel):
    built_in_primitives_enabled: bool = True
    temporal_primitives_enabled: bool = True
    timezone: str = "Europe/Berlin"
    reference_now: str | None = None
    custom_primitives: list[CustomPrimitiveRule] = Field(default_factory=list)
    session_memory_dataset_key: str = "session_memory"
    max_chars: int = Field(default=2000, ge=1)
    overlap_chars: int = Field(default=200, ge=0)
    route_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    min_candidate_evidence: int = Field(default=3, ge=1)


class ExistingDatasetSummary(BaseModel):
    dataset_key: str
    display_name: str = ""
    semantic_description: str = ""
    usage_guidance: str = ""
    llm_summary: str | None = None
    capability_tags: list[str] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    retrieval_capabilities: list[str] = Field(default_factory=list)


class AnalyzedChunk(BaseModel):
    id: str
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzedPrimitive(BaseModel):
    id: str
    kind: str
    text: str
    char_start: int
    char_end: int
    confidence: float
    subkind: str | None = None
    source: str = "built_in"
    target_dataset_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMemoryWriteProposal(BaseModel):
    dataset_key: str
    item_id: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrimitiveWriteProposal(BaseModel):
    item_id: str
    kind: str
    raw_text: str
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetRouteCandidate(BaseModel):
    dataset_key: str
    score: float
    keyword_score: float
    vector_score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    primitive_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class DatasetCreationCandidate(BaseModel):
    label: str
    suggested_dataset_key: str | None = None
    evidence_count: int
    ready_to_create: bool
    reasons: list[str] = Field(default_factory=list)
    primitive_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class GraphEdgeProposal(BaseModel):
    source_type: str
    source_key: str
    target_type: str
    target_key: str
    edge_type: str
    description: str = ""


class IngestAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = "main"
    config: IngestAnalysisConfig = Field(default_factory=IngestAnalysisConfig)
    existing_datasets: list[ExistingDatasetSummary] = Field(default_factory=list)
    candidate_state: list[CandidateEvidence] = Field(default_factory=list)


class IngestAnalysisResult(BaseModel):
    session_id: str
    chunks: list[AnalyzedChunk]
    primitives: list[AnalyzedPrimitive]
    session_memory_writes: list[SessionMemoryWriteProposal]
    primitive_write_proposals: list[PrimitiveWriteProposal]
    dataset_routes: list[DatasetRouteCandidate]
    dataset_creation_candidates: list[DatasetCreationCandidate]
    graph_edges: list[GraphEdgeProposal]
    metadata: dict[str, Any] = Field(default_factory=dict)
