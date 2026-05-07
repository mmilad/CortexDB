"""Optional OpenAI-compatible LLM provider for derived ingest work."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app.schemas.derived import DerivedMemoryEnvelope

_PROVIDER_ENV = "CORTEXDB_LLM_PROVIDER"
_URL_ENV = "CORTEXDB_LLM_URL"
_MODEL_ENV = "CORTEXDB_LLM_MODEL"
_KEY_ENV = "CORTEXDB_LLM_API_KEY"
_SCHEMA_NAME = "cortexdb_derived_memory"


class LLMService:
    def __init__(self) -> None:
        self.provider = os.environ.get(_PROVIDER_ENV, "none").lower()
        self.url = os.environ.get(_URL_ENV, "").rstrip("/")
        self.model = os.environ.get(_MODEL_ENV, "")
        self.api_key = os.environ.get(_KEY_ENV)

    def is_enabled(self) -> bool:
        return self.provider == "api" and bool(self.url and self.model)

    def _chat_payload(self, text: str, *, structured_schema: bool) -> dict[str, Any]:
        schema = DerivedMemoryEnvelope.model_json_schema()
        prompt = (
            "Extract durable memory from this input for CortexDB, a retrieval middleware. "
            "Return only data that matches the provided schema. "
            "Use memories=[] when nothing durable exists. "
            "Each memory is a small item suitable for later retrieval; do not include chat filler. "
            "dataset_key must be lowercase snake_case and should start with derived_. "
            "Prefer specific dataset keys such as derived_decisions, derived_goals, "
            "derived_preferences, derived_constraints, derived_tasks, derived_knowledge. "
            "metadata must be flat or shallow JSON and must not include the full input.\n\n"
            "JSON Schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}\n\n"
            f"Input:\n{text}"
        )
        response_format: dict[str, Any]
        if structured_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": _SCHEMA_NAME,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract durable memory for CortexDB. "
                        "You output JSON only and must follow the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": response_format,
        }

    def extract_memory(self, text: str) -> dict[str, Any]:
        if not self.is_enabled():
            raise RuntimeError("No CORTEXDB_LLM_PROVIDER=api configuration is available.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.url}/v1/chat/completions",
                headers=headers,
                json=self._chat_payload(text, structured_schema=True),
            )
            if response.status_code in (400, 404, 422):
                response = client.post(
                    f"{self.url}/v1/chat/completions",
                    headers=headers,
                    json=self._chat_payload(text, structured_schema=False),
                )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        envelope = DerivedMemoryEnvelope.model_validate_json(content)
        return envelope.model_dump()


def get_llm_service() -> LLMService:
    return LLMService()
