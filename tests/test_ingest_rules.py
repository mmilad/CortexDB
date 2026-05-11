from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

from app.ingest.rules import load_ingest_analysis_config
from app.schemas.ingest_rules import IngestRulePackRecord
from app.store import SqliteStore


def test_store_roundtrips_ingest_rule_pack() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    store = SqliteStore(path)
    try:
        pack = IngestRulePackRecord(
            key="framework_knowledge",
            display_name="Framework Knowledge",
            primitive_rules=[
                {
                    "kind": "framework",
                    "pattern": r"\b(Mastra|LangChain)\b",
                    "target_dataset_key": "frameworks",
                }
            ],
        )
        store.upsert_ingest_rule_pack(pack.key, pack.model_dump(mode="json"))

        rows = store.list_ingest_rule_packs(active_only=True)
        assert len(rows) == 1
        assert rows[0]["key"] == "framework_knowledge"
        assert rows[0]["created_at"] is not None
    finally:
        store.close()
        os.unlink(path)


def test_active_rule_packs_compile_into_analyzer_config() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    store = SqliteStore(path)
    try:
        pack = IngestRulePackRecord(
            key="framework_knowledge",
            display_name="Framework Knowledge",
            primitive_rules=[
                {
                    "kind": "framework",
                    "pattern": r"\bMastra\b",
                    "target_dataset_key": "frameworks",
                    "confidence": 0.82,
                }
            ],
            aliases=[
                {
                    "canonical": "LangChain",
                    "aliases": ["lang chain"],
                    "kind": "framework_alias",
                    "target_dataset_key": "frameworks",
                }
            ],
        )
        store.upsert_ingest_rule_pack(pack.key, pack.model_dump(mode="json"))

        config = load_ingest_analysis_config(store)
        assert {rule.kind for rule in config.custom_primitives} == {"framework", "framework_alias"}
        assert all(rule.target_dataset_key == "frameworks" for rule in config.custom_primitives)
    finally:
        store.close()
        os.unlink(path)


def test_ingest_rule_pack_api_workflow() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    os.environ["CORTEXDB_DB_PATH"] = db_path
    os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"

    import app.embed.service as embed_mod
    import app.store.main as store_mod

    store_mod._store = None
    embed_mod._service = embed_mod.EmbeddingService()

    from app.main import app

    try:
        with TestClient(app) as client:
            dataset = {
                "dataset_key": "frameworks",
                "display_name": "Frameworks",
                "schema_version": "v1",
                "semantic_description": "Agent frameworks and RAG libraries.",
                "usage_guidance": "Use for framework mentions.",
                "retrieval_capabilities": ["keyword"],
            }
            assert client.post("/datasets", json=dataset).status_code == 200

            context = client.get("/ingest/rule-packs/context")
            assert context.status_code == 200
            context_body = context.json()
            assert "primitive_rules" in context_body["accepted_objects"][0]
            assert "domain_context" in context_body
            assert "proposal_checklist" in context_body
            assert "json_contract_hint" in context_body

            pack = {
                "key": "framework_knowledge",
                "display_name": "Framework Knowledge",
                "primitive_rules": [
                    {
                        "kind": "framework",
                        "pattern": r"\b(Mastra|LangChain)\b",
                        "target_dataset_key": "frameworks",
                        "confidence": 0.82,
                    }
                ],
                "routing_hints": [
                    {
                        "target_dataset_key": "frameworks",
                        "match_terms": ["agent framework"],
                        "primitive_kinds": ["framework"],
                    }
                ],
            }
            validation = client.post("/ingest/rule-packs/validate", json=pack)
            assert validation.status_code == 200
            assert validation.json()["accepted"] is True
            assert validation.json()["compiled_custom_primitive_count"] == 2

            created = client.post("/ingest/rule-packs", json=pack)
            assert created.status_code == 200
            assert created.json()["created_at"] is not None

            analyzed = client.post(
                "/ingest/analyze",
                json={
                    "text": "Compare Mastra as an agent framework.",
                    "config": {"max_chars": 100, "overlap_chars": 0},
                },
            )
            assert analyzed.status_code == 200
            body = analyzed.json()
            assert body["metadata"]["active_rule_pack_count"] == 1
            assert "framework" in {primitive["kind"] for primitive in body["primitives"]}
            assert body["dataset_routes"][0]["dataset_key"] == "frameworks"

            enriched = client.get("/ingest/rule-packs/context")
            assert enriched.status_code == 200
            enriched_body = enriched.json()
            assert enriched_body["domain_context"]["active_rule_packs"][0]["key"] == "framework_knowledge"
            primitive_kinds = {item["kind"] for item in enriched_body["domain_context"]["primitive_kinds"]}
            assert {"task", "decision", "framework", "routing_hint"} & primitive_kinds
            profiles = {profile["kind"] for profile in enriched_body["knowledge_type_profiles"]}
            assert {"framework", "decision", "place", "person"} <= profiles
    finally:
        store_mod.close_store()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def test_invalid_regex_rule_pack_is_rejected() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    os.environ["CORTEXDB_DB_PATH"] = db_path
    os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"

    import app.embed.service as embed_mod
    import app.store.main as store_mod

    store_mod._store = None
    embed_mod._service = embed_mod.EmbeddingService()

    from app.main import app

    try:
        with TestClient(app) as client:
            pack = {
                "key": "bad_regex",
                "display_name": "Bad Regex",
                "primitive_rules": [{"kind": "broken", "pattern": "["}],
            }
            validation = client.post("/ingest/rule-packs/validate", json=pack)
            assert validation.status_code == 200
            assert validation.json()["accepted"] is False

            created = client.post("/ingest/rule-packs", json=pack)
            assert created.status_code == 422
    finally:
        store_mod.close_store()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def test_rule_pack_context_exposes_domain_metadata_without_memory_text() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    os.environ["CORTEXDB_DB_PATH"] = db_path
    os.environ["CORTEXDB_EMBED_PROVIDER"] = "none"

    import app.embed.service as embed_mod
    import app.store.main as store_mod

    store_mod._store = None
    embed_mod._service = embed_mod.EmbeddingService()

    from app.main import app
    from app.store import get_store

    try:
        with TestClient(app) as client:
            for key, entity_type in [
                ("frameworks", "Framework"),
                ("decisions", "Decision"),
                ("places", "Place"),
                ("persons", "Person"),
            ]:
                payload = {
                    "dataset_key": key,
                    "display_name": key.title(),
                    "schema_version": "v1",
                    "semantic_description": f"{entity_type} knowledge.",
                    "usage_guidance": f"Route {entity_type.lower()} mentions here.",
                    "entity_types": [entity_type],
                    "capability_tags": [key, "ingest_rules"],
                    "filterable_fields": ["kind", "source"],
                    "retrieval_capabilities": ["keyword"],
                    "query_examples": [
                        {
                            "label": f"{key}_by_keyword",
                            "description": f"Find {entity_type.lower()} entries by keyword.",
                            "example_request": {"keyword_query": entity_type.lower()},
                        }
                    ],
                }
                assert client.post("/datasets", json=payload).status_code == 200

            relationship = {
                "id": "person_decision_edge",
                "source_type": "dataset",
                "source_key": "persons",
                "target_type": "dataset",
                "target_key": "decisions",
                "edge_type": "related",
                "description": "People can own or influence decisions.",
            }
            assert client.post("/relationships", json=relationship).status_code == 200

            store = get_store()
            store.insert_memory_item(
                {
                    "id": "private-memory-text",
                    "dataset_key": "frameworks",
                    "raw_text": "PRIVATE_MEMORY_TEXT_SHOULD_NOT_LEAK",
                    "metadata": {"kind": "framework"},
                }
            )
            pack = IngestRulePackRecord(
                key="person_knowledge",
                display_name="Person Knowledge",
                primitive_rules=[
                    {
                        "kind": "person",
                        "pattern": r"\bAlice Smith\b",
                        "target_dataset_key": "persons",
                    }
                ],
                aliases=[
                    {
                        "canonical": "Alice Smith",
                        "aliases": ["@alice"],
                        "kind": "person_alias",
                        "target_dataset_key": "persons",
                    }
                ],
                metadata_fields=[
                    {"field": "role", "description": "Person role.", "example_values": ["owner"]}
                ],
                examples=[{"label": "known_person", "text": "Alice Smith owns the decision."}],
            )
            assert client.post("/ingest/rule-packs", json=pack.model_dump(mode="json")).status_code == 200

            response = client.get("/ingest/rule-packs/context")
            assert response.status_code == 200
            body = response.json()
            dataset_keys = {dataset["dataset_key"] for dataset in body["domain_context"]["datasets"]}
            assert {"frameworks", "decisions", "places", "persons"} <= dataset_keys

            persons = next(dataset for dataset in body["domain_context"]["datasets"] if dataset["dataset_key"] == "persons")
            assert persons["entity_types"] == ["Person"]
            assert persons["query_examples"][0]["label"] == "persons_by_keyword"

            rels = body["domain_context"]["relationships"]
            assert any(rel["source_key"] == "persons" and rel["target_key"] == "decisions" for rel in rels)

            active_pack = body["domain_context"]["active_rule_packs"][0]
            assert active_pack["key"] == "person_knowledge"
            assert active_pack["primitive_kinds"] == ["person"]
            assert active_pack["alias_kinds"] == ["person_alias"]
            assert active_pack["routing_targets"] == ["persons"]
            assert active_pack["metadata_fields"] == ["role"]
            assert active_pack["example_labels"] == ["known_person"]

            primitive_kinds = {item["kind"]: item for item in body["domain_context"]["primitive_kinds"]}
            assert primitive_kinds["person"]["source"] == "rule_pack"
            assert primitive_kinds["person"]["routing_targets"] == ["persons"]
            assert "PRIVATE_MEMORY_TEXT_SHOULD_NOT_LEAK" not in response.text
    finally:
        store_mod.close_store()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
