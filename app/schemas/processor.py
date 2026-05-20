"""Schemas for optional text processor sidecars."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ProcessorProvider = Literal["none", "sidecar", "local"]
ProcessorStrategy = Literal["fallback", "safe", "semantic", "extractive"]
SidecarStrategy = Literal["safe", "semantic", "extractive"]
StructuralPrimitive = Literal["document", "section", "paragraph", "sentence", "span", "chunk"]
MeaningPrimitive = Literal[
    "fact",
    "decision",
    "task",
    "goal",
    "constraint",
    "preference",
    "definition",
    "entity",
    "relationship",
    "event",
    "question",
    "answer",
    "code_symbol",
    "error",
    "command",
    "reference",
]


class ProcessorSpan(BaseModel):
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    primitive: StructuralPrimitive = "chunk"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorPrimitive(BaseModel):
    kind: str
    subkind: str | None = None
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorEntity(BaseModel):
    text: str
    label: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorPhrase(BaseModel):
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    label: str = "noun_phrase"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorClassification(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    matched_rule_key: str | None = None
    target_dataset_key: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorCandidate(BaseModel):
    label: str
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    suggested_dataset_key: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ProcessorRequest(BaseModel):
    text: str = Field(..., min_length=1)
    strategy: SidecarStrategy = "safe"
    max_chars: int = Field(default=2000, ge=1)
    overlap_chars: int = Field(default=200, ge=0)
    extract_primitives: bool = False
    classify: bool = False
    known_match_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    candidate_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorResponse(BaseModel):
    processor: str
    processor_version: str
    strategy: SidecarStrategy
    chunks: list[ProcessorSpan] = Field(default_factory=list)
    primitives: list[ProcessorPrimitive] = Field(default_factory=list)
    entities: list[ProcessorEntity] = Field(default_factory=list)
    phrases: list[ProcessorPhrase] = Field(default_factory=list)
    classifications: list[ProcessorClassification] = Field(default_factory=list)
    candidates: list[ProcessorCandidate] = Field(default_factory=list)


class ProcessorJobResult(BaseModel):
    name: str = "processor"
    status: Literal["completed", "skipped", "failed"]
    detail: str = ""
    strategy: ProcessorStrategy = "fallback"
    primitive_count: int = 0
