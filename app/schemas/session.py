"""Schemas for session-aware ingest and context middleware."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SessionType = Literal["chat", "project", "task", "custom"]
ScopeMode = Literal["namespace", "global", "explicit"]
DatasetPolicy = Literal["create_if_needed", "use_existing", "never_create", "explicit_only"]
MessageRole = Literal["system", "user", "assistant", "tool", "event"]
DerivedStatus = Literal["completed", "partial", "skipped", "failed"]


class SessionRecord(BaseModel):
    id: str = "main"
    type: SessionType = "chat"
    scope_mode: ScopeMode = "namespace"
    namespace: str | None = None
    dataset_policy: DatasetPolicy = "create_if_needed"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class RawTextRecord(BaseModel):
    id: str
    text: str
    source: str = "unknown"
    relations: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    created_at: str | None = None


class SessionMessageRecord(BaseModel):
    id: str
    session_id: str
    role: MessageRole = "user"
    content: str
    raw_text_id: str | None = None
    token_estimate: int = 0
    autocontext_enabled: bool = True
    summary_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class SessionSummaryRecord(BaseModel):
    id: str
    session_id: str
    summary: str
    message_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = "main"
    session_type: SessionType = "chat"
    role: MessageRole = "user"
    source: str = "user_prompt"
    scope_mode: ScopeMode = "namespace"
    namespace: str | None = None
    dataset_policy: DatasetPolicy = "create_if_needed"
    dataset_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_relations: dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = Field(default=4000, ge=1)
    summary_target_tokens: int = Field(default=800, ge=1)
    derive: bool = True


class DerivedJobResult(BaseModel):
    name: str
    status: DerivedStatus
    detail: str = ""
    dataset_keys: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)


class IngestRouteTargetTrace(BaseModel):
    dataset_key: str
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    primitive_count: int = 0
    chunk_count: int = 0


class IngestCanonicalEntityTrace(BaseModel):
    id: str
    dataset_key: str
    entity_kind: str
    name: str
    observation_id: str
    mention_count: int = 1


class IngestTrace(BaseModel):
    status: DerivedStatus = "completed"
    chunks_written: int = 0
    observations_written: int = 0
    canonical_entities_upserted: int = 0
    candidate_observations_written: int = 0
    graph_edges_written: int = 0
    route_targets: list[IngestRouteTargetTrace] = Field(default_factory=list)
    observation_kinds: dict[str, int] = Field(default_factory=dict)
    canonical_entities: list[IngestCanonicalEntityTrace] = Field(default_factory=list)
    candidate_labels: list[str] = Field(default_factory=list)


class IngestResult(BaseModel):
    session: SessionRecord
    message: SessionMessageRecord
    raw_text: RawTextRecord
    derived: list[DerivedJobResult]
    trace: IngestTrace | None = None


class ContextRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: str = "main"
    top_k: int = Field(default=5, ge=1, le=50)
    dataset_keys: list[str] = Field(default_factory=list)


class ContextItem(BaseModel):
    kind: str
    text: str
    source_id: str | None = None
    dataset_key: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackage(BaseModel):
    session: SessionRecord
    prompt: str
    summaries: list[SessionSummaryRecord]
    messages: list[SessionMessageRecord]
    items: list[ContextItem]
    dataset_keys: list[str]
    usage_hint: str
