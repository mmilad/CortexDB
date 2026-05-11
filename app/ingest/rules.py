"""Schema-driven ingest rule pack helpers."""

from __future__ import annotations

import re

from app.schemas.ingest_analysis import CustomPrimitiveRule, IngestAnalysisConfig
from app.schemas.ingest_rules import (
    IngestAliasRuleSpec,
    IngestKnowledgeTypeProfile,
    IngestMetadataFieldSpec,
    IngestPrimitiveKindSummary,
    IngestRuleDatasetSummary,
    IngestRuleDomainContext,
    IngestRuleGuidance,
    IngestRulePackSummary,
    IngestRulePackRecord,
    IngestRuleRelationshipSummary,
    IngestRulePackValidationResult,
)
from app.store import SqliteStore

BUILT_IN_PRIMITIVE_KINDS = [
    "task",
    "decision",
    "constraint",
    "preference",
    "fact",
    "event",
    "error",
    "command",
    "reference",
    "entity",
    "time",
]


def _alias_pattern(alias: IngestAliasRuleSpec) -> str:
    terms = [alias.canonical, *alias.aliases]
    escaped = [re.escape(term) for term in terms if term.strip()]
    return r"\b(" + "|".join(escaped) + r")\b"


def _routing_pattern(terms: list[str]) -> str:
    escaped = [re.escape(term) for term in terms if term.strip()]
    return r"\b(" + "|".join(escaped) + r")\b"


def compile_rule_pack_primitives(pack: IngestRulePackRecord) -> list[CustomPrimitiveRule]:
    """Compile a stored rule pack into analyzer custom primitive rules."""
    rules: list[CustomPrimitiveRule] = []
    for primitive in pack.primitive_rules:
        rules.append(
            CustomPrimitiveRule(
                kind=primitive.kind,
                subkind=primitive.subkind,
                pattern=primitive.pattern,
                target_dataset_key=primitive.target_dataset_key,
                confidence=primitive.confidence,
                metadata={
                    **primitive.metadata,
                    "rule_pack_key": pack.key,
                    "rule_pack_object": "primitive_rule",
                },
            )
        )

    for alias in pack.aliases:
        if not alias.canonical.strip() and not alias.aliases:
            continue
        rules.append(
            CustomPrimitiveRule(
                kind=alias.kind,
                subkind="alias",
                pattern=_alias_pattern(alias),
                target_dataset_key=alias.target_dataset_key,
                confidence=alias.confidence,
                metadata={
                    **alias.metadata,
                    "canonical": alias.canonical,
                    "aliases": alias.aliases,
                    "rule_pack_key": pack.key,
                    "rule_pack_object": "alias",
                },
            )
        )

    for hint in pack.routing_hints:
        if not hint.match_terms:
            continue
        rules.append(
            CustomPrimitiveRule(
                kind="routing_hint",
                subkind="dataset_route",
                pattern=_routing_pattern(hint.match_terms),
                target_dataset_key=hint.target_dataset_key,
                confidence=hint.confidence,
                metadata={
                    "rule_pack_key": pack.key,
                    "rule_pack_object": "routing_hint",
                    "description": hint.description,
                    "primitive_kinds": hint.primitive_kinds,
                },
            )
        )
    return rules


def validate_rule_pack(
    pack: IngestRulePackRecord,
    *,
    known_dataset_keys: set[str] | None = None,
) -> IngestRulePackValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    known_dataset_keys = known_dataset_keys or set()

    for index, primitive in enumerate(pack.primitive_rules):
        try:
            re.compile(primitive.pattern)
        except re.error as exc:
            errors.append(f"primitive_rules[{index}].pattern is not valid regex: {exc}")
        if primitive.target_dataset_key and known_dataset_keys and primitive.target_dataset_key not in known_dataset_keys:
            warnings.append(
                f"primitive_rules[{index}].target_dataset_key '{primitive.target_dataset_key}' is not registered"
            )

    for index, alias in enumerate(pack.aliases):
        if not alias.canonical.strip():
            errors.append(f"aliases[{index}].canonical must not be blank")
        if not alias.aliases:
            warnings.append(f"aliases[{index}] has no aliases; only the canonical term will match")
        if alias.target_dataset_key and known_dataset_keys and alias.target_dataset_key not in known_dataset_keys:
            warnings.append(f"aliases[{index}].target_dataset_key '{alias.target_dataset_key}' is not registered")

    for index, hint in enumerate(pack.routing_hints):
        if not hint.match_terms and not hint.primitive_kinds:
            warnings.append(f"routing_hints[{index}] has no match_terms or primitive_kinds")
        if known_dataset_keys and hint.target_dataset_key not in known_dataset_keys:
            warnings.append(f"routing_hints[{index}].target_dataset_key '{hint.target_dataset_key}' is not registered")

    compiled_count = 0
    if not errors:
        try:
            compiled_rules = compile_rule_pack_primitives(pack)
            for rule in compiled_rules:
                re.compile(rule.pattern)
            compiled_count = len(compiled_rules)
        except re.error as exc:
            errors.append(f"compiled primitive pattern is not valid regex: {exc}")

    return IngestRulePackValidationResult(
        accepted=not errors,
        errors=errors,
        warnings=warnings,
        compiled_custom_primitive_count=compiled_count,
    )


def load_ingest_analysis_config(
    store: SqliteStore,
    *,
    namespace: str | None = None,
    base_config: IngestAnalysisConfig | None = None,
) -> IngestAnalysisConfig:
    config = (base_config or IngestAnalysisConfig()).model_copy(deep=True)
    active_packs = [
        IngestRulePackRecord.model_validate(row)
        for row in store.list_ingest_rule_packs(namespace=namespace, active_only=True)
    ]
    custom = list(config.custom_primitives)
    for pack in active_packs:
        custom.extend(compile_rule_pack_primitives(pack))
    return config.model_copy(update={"custom_primitives": custom})


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dataset_summary(data: dict) -> IngestRuleDatasetSummary:
    return IngestRuleDatasetSummary(
        dataset_key=data.get("dataset_key") or data.get("key") or "",
        display_name=data.get("display_name", ""),
        semantic_description=data.get("semantic_description", ""),
        usage_guidance=data.get("usage_guidance", ""),
        entity_types=list(data.get("entity_types", [])),
        capability_tags=list(data.get("capability_tags", [])),
        filterable_fields=list(data.get("filterable_fields", [])),
        query_examples=list(data.get("query_examples", [])),
        status=data.get("status", "active"),
    )


def _relationship_summary(data: dict) -> IngestRuleRelationshipSummary:
    return IngestRuleRelationshipSummary(
        source_type=data.get("source_type", ""),
        source_key=data.get("source_key", ""),
        target_type=data.get("target_type", ""),
        target_key=data.get("target_key", ""),
        edge_type=data.get("edge_type", ""),
        description=data.get("description", ""),
        join_fields=list(data.get("join_fields", [])),
    )


def _pack_summary(pack: IngestRulePackRecord) -> IngestRulePackSummary:
    routing_targets = [
        primitive.target_dataset_key
        for primitive in pack.primitive_rules
        if primitive.target_dataset_key
    ]
    routing_targets.extend(alias.target_dataset_key for alias in pack.aliases if alias.target_dataset_key)
    routing_targets.extend(hint.target_dataset_key for hint in pack.routing_hints)
    return IngestRulePackSummary(
        key=pack.key,
        display_name=pack.display_name,
        status=pack.status,
        primitive_kinds=_unique([primitive.kind for primitive in pack.primitive_rules]),
        alias_kinds=_unique([alias.kind for alias in pack.aliases]),
        routing_targets=_unique(routing_targets),
        metadata_fields=_unique([field.field for field in pack.metadata_fields]),
        example_labels=_unique([example.label for example in pack.examples]),
    )


def _primitive_kind_summaries(active_packs: list[IngestRulePackRecord]) -> list[IngestPrimitiveKindSummary]:
    summaries = [
        IngestPrimitiveKindSummary(kind=kind, source="built_in")
        for kind in BUILT_IN_PRIMITIVE_KINDS
    ]
    by_kind: dict[str, IngestPrimitiveKindSummary] = {}
    for pack in active_packs:
        for primitive in pack.primitive_rules:
            summary = by_kind.setdefault(
                primitive.kind,
                IngestPrimitiveKindSummary(kind=primitive.kind, source="rule_pack"),
            )
            summary.rule_pack_keys.append(pack.key)
            if primitive.target_dataset_key:
                summary.routing_targets.append(primitive.target_dataset_key)
        for alias in pack.aliases:
            summary = by_kind.setdefault(
                alias.kind,
                IngestPrimitiveKindSummary(kind=alias.kind, source="rule_pack"),
            )
            summary.rule_pack_keys.append(pack.key)
            if alias.target_dataset_key:
                summary.routing_targets.append(alias.target_dataset_key)
        for hint in pack.routing_hints:
            summary = by_kind.setdefault(
                "routing_hint",
                IngestPrimitiveKindSummary(kind="routing_hint", source="rule_pack"),
            )
            summary.rule_pack_keys.append(pack.key)
            summary.routing_targets.append(hint.target_dataset_key)
    for summary in by_kind.values():
        summary.rule_pack_keys = _unique(summary.rule_pack_keys)
        summary.routing_targets = _unique(summary.routing_targets)
    summaries.extend(sorted(by_kind.values(), key=lambda item: item.kind))
    return summaries


def _knowledge_type_profiles() -> list[IngestKnowledgeTypeProfile]:
    return [
        IngestKnowledgeTypeProfile(
            kind="framework",
            purpose="Identify software frameworks, libraries, SDKs, and platform ecosystems.",
            alias_guidance="Use canonical product names plus common abbreviations or spelling variants.",
            regex_guidance="Prefer explicit bounded alternations for known framework names.",
            suggested_metadata_fields=[
                IngestMetadataFieldSpec(field="domain", description="Domain bucket.", example_values=["agent_frameworks"]),
                IngestMetadataFieldSpec(field="vendor", description="Owning company or project.", example_values=["OpenAI"]),
                IngestMetadataFieldSpec(field="ecosystem", description="Technical ecosystem.", example_values=["python", "typescript"]),
            ],
            routing_guidance="Route to datasets describing frameworks, libraries, tools, or architecture comparisons.",
            example_cues=["LangChain", "Mastra", "framework", "SDK", "library"],
        ),
        IngestKnowledgeTypeProfile(
            kind="decision",
            purpose="Capture decisions, accepted/rejected options, and tradeoffs.",
            alias_guidance="Aliases are usually cue phrases rather than entity aliases.",
            regex_guidance="Use decision verbs and phrases such as decided, accepted, rejected, go with, choose, tradeoff.",
            suggested_metadata_fields=[
                IngestMetadataFieldSpec(field="status", description="Decision status.", example_values=["accepted", "rejected"]),
                IngestMetadataFieldSpec(field="rationale", description="Reason or tradeoff summary."),
                IngestMetadataFieldSpec(field="scope", description="Project or component affected."),
            ],
            routing_guidance="Route to decision, architecture, planning, or project-memory datasets.",
            example_cues=["we decided", "accepted", "rejected", "tradeoff", "go with"],
        ),
        IngestKnowledgeTypeProfile(
            kind="place",
            purpose="Identify physical places, regions, offices, venues, and geographic references.",
            alias_guidance="Use conservative aliases for known places; avoid broad patterns that capture arbitrary capitalized words.",
            regex_guidance="Prefer curated place lists or bounded patterns with place-specific context words.",
            suggested_metadata_fields=[
                IngestMetadataFieldSpec(field="place_type", description="Kind of place.", example_values=["city", "office"]),
                IngestMetadataFieldSpec(field="country", description="Country or jurisdiction."),
                IngestMetadataFieldSpec(field="region", description="State, province, or region."),
            ],
            routing_guidance="Route to location, travel, CRM, venue, or geography datasets.",
            example_cues=["Berlin", "office", "venue", "city", "region"],
        ),
        IngestKnowledgeTypeProfile(
            kind="person",
            purpose="Identify known people, contacts, owners, authors, or stakeholders.",
            alias_guidance="Use canonical full names with known nicknames, handles, or initials when provided.",
            regex_guidance="Prefer known-name alternations; generic person-name regexes should remain low confidence.",
            suggested_metadata_fields=[
                IngestMetadataFieldSpec(field="canonical_name", description="Stable full name."),
                IngestMetadataFieldSpec(field="role", description="Role or responsibility.", example_values=["maintainer"]),
                IngestMetadataFieldSpec(field="organization", description="Affiliated team or company."),
            ],
            routing_guidance="Route to people, contacts, team, authorship, or stakeholder datasets.",
            example_cues=["Alice Smith", "@alice", "owner", "maintainer", "stakeholder"],
        ),
    ]


def _domain_context(store: SqliteStore, active_packs: list[IngestRulePackRecord]) -> IngestRuleDomainContext:
    datasets = [
        _dataset_summary(data)
        for _, data in sorted(store.list_datasets().items())
    ]
    relationships = [
        _relationship_summary(data)
        for data in store.list_relationships()
    ]
    return IngestRuleDomainContext(
        datasets=datasets,
        relationships=relationships,
        active_rule_packs=[_pack_summary(pack) for pack in active_packs],
        primitive_kinds=_primitive_kind_summaries(active_packs),
    )


def build_rule_guidance(store: SqliteStore, *, namespace: str | None = None) -> IngestRuleGuidance:
    active_packs = [
        IngestRulePackRecord.model_validate(row)
        for row in store.list_ingest_rule_packs(namespace=namespace, active_only=True)
    ]
    return IngestRuleGuidance(
        purpose=(
            "Teach CortexDB deterministic ingest analysis rules without code changes. "
            "CortexDB validates and stores schema-driven rule packs; caller LLMs own all reasoning."
        ),
        workflow=[
            "Read this guidance before proposing a new knowledge type.",
            "Return an IngestRulePackRecord JSON object with regexes, aliases, routing hints, examples, and notes.",
            "POST the object to /ingest/rule-packs/validate for dry-run validation.",
            "POST the accepted object to /ingest/rule-packs to persist it.",
            "Call POST /ingest/analyze to preview how active rule packs affect analyzer proposals.",
        ],
        accepted_objects=[
            "primitive_rules: regex extractors compiled into analyzer custom primitives",
            "aliases: canonical names plus alternate surface forms compiled into regex primitives",
            "relationship_patterns: validated and stored graph templates for future promotion workflows",
            "routing_hints: dataset targets plus match terms compiled into route-oriented primitives",
            "metadata_fields: expected metadata glossary for client-side structured writes",
            "examples: natural-language fixtures an LLM can use to self-check a proposal",
        ],
        built_in_primitive_kinds=BUILT_IN_PRIMITIVE_KINDS,
        naming_conventions=[
            "Use lowercase snake_case keys and primitive kinds.",
            "Keep regexes deterministic and bounded; avoid patterns that can match empty text.",
            "Use target_dataset_key when matches should force a route candidate.",
            "Store explanatory notes in validation_notes instead of relying on hidden prompt context.",
        ],
        validation_rules=[
            "Pydantic validates required fields, naming shape, confidence range, and object types.",
            "CortexDB compiles every regex before accepting a pack.",
            "Unknown target datasets produce warnings, not hard errors, so datasets can be created later.",
            "No generative LLM calls are made inside CortexDB.",
        ],
        example_proposal=IngestRulePackRecord(
            key="framework_knowledge",
            display_name="Framework Knowledge",
            description="Extract framework mentions and route them to framework notes.",
            primitive_rules=[
                {
                    "kind": "framework",
                    "pattern": r"\b(Mastra|LangChain|LlamaIndex|Haystack)\b",
                    "target_dataset_key": "frameworks",
                    "confidence": 0.82,
                    "metadata": {"domain": "agent_frameworks"},
                }
            ],
            aliases=[
                {
                    "canonical": "LangChain",
                    "aliases": ["lang chain", "LCEL"],
                    "kind": "framework_alias",
                    "target_dataset_key": "frameworks",
                }
            ],
            routing_hints=[
                {
                    "target_dataset_key": "frameworks",
                    "match_terms": ["agent framework", "RAG library"],
                    "primitive_kinds": ["framework", "framework_alias"],
                }
            ],
            metadata_fields=[
                {
                    "field": "domain",
                    "description": "Domain bucket for the extracted primitive.",
                    "example_values": ["agent_frameworks"],
                }
            ],
            examples=[
                {
                    "label": "framework_mentions",
                    "text": "Compare Mastra and LangChain for RAG routing.",
                    "expected_primitives": [{"kind": "framework", "texts": ["Mastra", "LangChain"]}],
                }
            ],
            validation_notes=["Regex should only match concrete framework names."],
        ),
        active_rule_packs=active_packs,
        domain_context=_domain_context(store, active_packs),
        knowledge_type_profiles=_knowledge_type_profiles(),
        proposal_checklist=[
            "Choose a lowercase snake_case rule-pack key and primitive kind names.",
            "Reuse an existing dataset_key from domain_context.datasets when routing is known.",
            "Check domain_context.primitive_kinds before inventing a new primitive kind.",
            "Provide regex primitive_rules or aliases that are deterministic and bounded.",
            "Add routing_hints when text cues should point to a target dataset even without entity matches.",
            "Declare metadata_fields that clients should attach to future primitive writes.",
            "Include examples with expected_primitives so the proposal can be reviewed and dry-run.",
            "Call /ingest/rule-packs/validate before persisting the pack.",
        ],
        json_contract_hint=(
            "Return one JSON object matching IngestRulePackRecord: key, display_name, optional description, "
            "status, primitive_rules, aliases, relationship_patterns, routing_hints, metadata_fields, "
            "examples, validation_notes, and metadata. Do not include prose outside JSON when posting."
        ),
    )
