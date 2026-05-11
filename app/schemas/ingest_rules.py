from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RulePackStatus = Literal["draft", "active", "disabled"]


class IngestPrimitiveRuleSpec(BaseModel):
    kind: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    pattern: str = Field(..., min_length=1)
    subkind: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    target_dataset_key: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class IngestAliasRuleSpec(BaseModel):
    canonical: str = Field(..., min_length=1)
    aliases: list[str] = Field(default_factory=list)
    kind: str = Field(default="alias", min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    target_dataset_key: str | None = None
    confidence: float = Field(default=0.68, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def aliases_must_not_be_empty_strings(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class IngestRelationshipPatternSpec(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    source_kind: str = Field(..., min_length=1)
    target_kind: str = Field(..., min_length=1)
    edge_type: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRoutingHintSpec(BaseModel):
    target_dataset_key: str = Field(..., min_length=1)
    match_terms: list[str] = Field(default_factory=list)
    primitive_kinds: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    description: str = ""

    @field_validator("match_terms", "primitive_kinds")
    @classmethod
    def strip_empty_values(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class IngestMetadataFieldSpec(BaseModel):
    field: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    value_type: str = Field(default="string")
    example_values: list[str] = Field(default_factory=list)


class IngestRuleExample(BaseModel):
    label: str
    text: str
    expected_primitives: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""


class IngestRulePackRecord(BaseModel):
    key: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    description: str = ""
    namespace: str | None = None
    status: RulePackStatus = "active"
    primitive_rules: list[IngestPrimitiveRuleSpec] = Field(default_factory=list)
    aliases: list[IngestAliasRuleSpec] = Field(default_factory=list)
    relationship_patterns: list[IngestRelationshipPatternSpec] = Field(default_factory=list)
    routing_hints: list[IngestRoutingHintSpec] = Field(default_factory=list)
    metadata_fields: list[IngestMetadataFieldSpec] = Field(default_factory=list)
    examples: list[IngestRuleExample] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class IngestRulePackValidationResult(BaseModel):
    accepted: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    compiled_custom_primitive_count: int = 0
    usage_hint: str = (
        "If accepted is true, POST the same rule pack to /ingest/rule-packs "
        "to persist it. Active packs are used by POST /ingest/analyze."
    )


class IngestRuleDatasetSummary(BaseModel):
    dataset_key: str
    display_name: str = ""
    semantic_description: str = ""
    usage_guidance: str = ""
    entity_types: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    filterable_fields: list[str] = Field(default_factory=list)
    query_examples: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "active"


class IngestRuleRelationshipSummary(BaseModel):
    source_type: str
    source_key: str
    target_type: str
    target_key: str
    edge_type: str
    description: str = ""
    join_fields: list[str] = Field(default_factory=list)


class IngestRulePackSummary(BaseModel):
    key: str
    display_name: str
    status: RulePackStatus
    primitive_kinds: list[str] = Field(default_factory=list)
    alias_kinds: list[str] = Field(default_factory=list)
    routing_targets: list[str] = Field(default_factory=list)
    metadata_fields: list[str] = Field(default_factory=list)
    example_labels: list[str] = Field(default_factory=list)


class IngestPrimitiveKindSummary(BaseModel):
    kind: str
    source: Literal["built_in", "rule_pack"]
    rule_pack_keys: list[str] = Field(default_factory=list)
    routing_targets: list[str] = Field(default_factory=list)


class IngestRuleDomainContext(BaseModel):
    datasets: list[IngestRuleDatasetSummary] = Field(default_factory=list)
    relationships: list[IngestRuleRelationshipSummary] = Field(default_factory=list)
    active_rule_packs: list[IngestRulePackSummary] = Field(default_factory=list)
    primitive_kinds: list[IngestPrimitiveKindSummary] = Field(default_factory=list)
    usage_hint: str = (
        "Use datasets as routing targets, relationships as graph-shape hints, "
        "active_rule_packs to avoid duplicates, and primitive_kinds to reuse established names."
    )


class IngestKnowledgeTypeProfile(BaseModel):
    kind: str
    purpose: str
    alias_guidance: str
    regex_guidance: str
    suggested_metadata_fields: list[IngestMetadataFieldSpec] = Field(default_factory=list)
    routing_guidance: str
    example_cues: list[str] = Field(default_factory=list)


class IngestRuleGuidance(BaseModel):
    purpose: str
    workflow: list[str]
    accepted_objects: list[str]
    built_in_primitive_kinds: list[str]
    naming_conventions: list[str]
    validation_rules: list[str]
    example_proposal: IngestRulePackRecord
    active_rule_packs: list[IngestRulePackRecord]
    domain_context: IngestRuleDomainContext = Field(default_factory=IngestRuleDomainContext)
    knowledge_type_profiles: list[IngestKnowledgeTypeProfile] = Field(default_factory=list)
    proposal_checklist: list[str] = Field(default_factory=list)
    json_contract_hint: str = ""
