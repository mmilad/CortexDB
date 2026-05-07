from __future__ import annotations

from app.llm.service import LLMService
from app.schemas.derived import DerivedMemoryEnvelope
from app.services.derived import run_derived_workflow
from app.store import SqliteStore


class FakeGenericLLM:
    def is_enabled(self) -> bool:
        return True

    def extract_memory(self, text: str) -> dict:
        return {
            "schema_version": "cortexdb.derived_memory.v1",
            "memories": [
                {
                    "dataset_key": "derived_preferences",
                    "kind": "preference",
                    "text": "User prefers local Ollama models for development.",
                    "score": 0.9,
                    "metadata": {"scope": "dev"},
                    "dataset": {
                        "display_name": "Derived Preferences",
                        "semantic_description": "User and workflow preferences extracted from ingest.",
                        "usage_guidance": "Use when adapting assistant behavior to user preferences.",
                        "entity_types": ["Preference"],
                        "capability_tags": ["derived", "preferences"],
                    },
                }
            ],
        }


class FakeLegacyLLM:
    def is_enabled(self) -> bool:
        return True

    def extract_memory(self, text: str) -> dict:
        return {"decisions": [{"text": "Keep /ingest as the front door.", "score": 1.0}]}


def test_derived_memory_schema_accepts_generic_envelope() -> None:
    envelope = DerivedMemoryEnvelope.model_validate(
        {
            "schema_version": "cortexdb.derived_memory.v1",
            "memories": [
                {
                    "dataset_key": "derived_constraints",
                    "kind": "constraint",
                    "text": "CortexDB should not perform final assistant reasoning.",
                    "score": 0.95,
                    "metadata": {"source": "test"},
                }
            ],
        }
    )

    assert envelope.memories[0].dataset_key == "derived_constraints"
    assert envelope.memories[0].kind == "constraint"


def test_llm_payload_uses_json_schema_response_format(monkeypatch) -> None:
    monkeypatch.setenv("CORTEXDB_LLM_PROVIDER", "api")
    monkeypatch.setenv("CORTEXDB_LLM_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("CORTEXDB_LLM_MODEL", "llama3")
    svc = LLMService()

    payload = svc._chat_payload("remember this", structured_schema=True)

    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["memories"]["type"] == "array"


def test_derived_workflow_writes_generic_dataset(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "test.sqlite"))
    try:
        result = run_derived_workflow(
            store=store,
            text="preference",
            dataset_policy="create_if_needed",
            dataset_keys=[],
            derive=True,
            llm_svc=FakeGenericLLM(),  # type: ignore[arg-type]
            session_id="s1",
            raw_text_id="r1",
        )

        assert result[0].status == "completed"
        assert result[0].dataset_keys == ["derived_preferences"]
        assert store.get_dataset("derived_preferences") is not None
        items = store.list_memory_items("derived_preferences")
        assert items[0]["raw_text"] == "User prefers local Ollama models for development."
        assert items[0]["metadata"]["derived_kind"] == "preference"
    finally:
        store.close()


def test_derived_workflow_keeps_legacy_bucket_support(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "test.sqlite"))
    try:
        result = run_derived_workflow(
            store=store,
            text="decision",
            dataset_policy="create_if_needed",
            dataset_keys=[],
            derive=True,
            llm_svc=FakeLegacyLLM(),  # type: ignore[arg-type]
            session_id="s1",
            raw_text_id="r1",
        )

        assert result[0].status == "completed"
        assert result[0].dataset_keys == ["derived_decisions"]
        items = store.list_memory_items("derived_decisions")
        assert items[0]["raw_text"] == "Keep /ingest as the front door."
    finally:
        store.close()
