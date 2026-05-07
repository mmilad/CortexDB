"""Optional OpenAI-compatible LLM provider for derived ingest work."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

_PROVIDER_ENV = "CORTEXDB_LLM_PROVIDER"
_URL_ENV = "CORTEXDB_LLM_URL"
_MODEL_ENV = "CORTEXDB_LLM_MODEL"
_KEY_ENV = "CORTEXDB_LLM_API_KEY"


class LLMService:
    def __init__(self) -> None:
        self.provider = os.environ.get(_PROVIDER_ENV, "none").lower()
        self.url = os.environ.get(_URL_ENV, "").rstrip("/")
        self.model = os.environ.get(_MODEL_ENV, "")
        self.api_key = os.environ.get(_KEY_ENV)

    def is_enabled(self) -> bool:
        return self.provider == "api" and bool(self.url and self.model)

    def extract_memory(self, text: str) -> dict[str, list[dict[str, Any]]]:
        if not self.is_enabled():
            raise RuntimeError("No CORTEXDB_LLM_PROVIDER=api configuration is available.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt = (
            "Extract durable memory from this input for a retrieval middleware. "
            "Return only JSON with keys facts, decisions, goals, knowledge. "
            "Each key must contain a list of objects with text, score, and metadata. "
            "Use empty lists when nothing durable exists.\n\n"
            f"Input:\n{text}"
        )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {
            "facts": list(parsed.get("facts", [])),
            "decisions": list(parsed.get("decisions", [])),
            "goals": list(parsed.get("goals", [])),
            "knowledge": list(parsed.get("knowledge", [])),
        }


def get_llm_service() -> LLMService:
    return LLMService()
