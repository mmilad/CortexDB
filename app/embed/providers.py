"""Embedding provider implementations.

All providers implement the same interface:
    embed(texts: list[str]) -> list[list[float]]

Callers never touch vectors — only raw text goes in, vectors come out.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from app.embed.config import EmbedConfig

logger = logging.getLogger("cortexdb.embed")


class EmbeddingProvider(ABC):
    """Base protocol for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input, in order."""

    @abstractmethod
    def model_id(self) -> str:
        """Return a stable identifier: 'provider/model' used for provenance tracking."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class NullProvider(EmbeddingProvider):
    """Used when CORTEXDB_EMBED_PROVIDER=none. Raises on any embed call."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "Embedding is disabled (CORTEXDB_EMBED_PROVIDER=none). "
            "Set a provider to enable vector search."
        )

    def model_id(self) -> str:
        return "none/none"


class OllamaProvider(EmbeddingProvider):
    """Calls the Ollama /api/embed endpoint (Ollama >= 0.3).

    Falls back to /api/embeddings (single-text, older API) if the batch
    endpoint is not available.
    """

    def __init__(self, config: EmbedConfig, client: httpx.Client | None = None) -> None:
        self._url = config.url
        self._model = config.model
        self._client = client or httpx.Client(timeout=60.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Try batch endpoint first (Ollama >= 0.3)
        try:
            resp = self._client.post(
                f"{self._url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            if resp.status_code == 200:
                data = resp.json()
                # Ollama returns {"embeddings": [[...], ...]}
                if "embeddings" in data:
                    return data["embeddings"]
        except httpx.RequestError:
            pass

        # Fallback: single-text legacy endpoint
        results = []
        for text in texts:
            resp = self._client.post(
                f"{self._url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results

    def model_id(self) -> str:
        return f"ollama/{self._model}"

    def ensure_model_pulled(self) -> None:
        """Pull the model if not already present. Blocks until done."""
        logger.info("Checking Ollama model %s...", self._model)
        resp = self._client.post(
            f"{self._url}/api/pull",
            json={"name": self._model, "stream": False},
            timeout=300.0,
        )
        resp.raise_for_status()
        logger.info("Model %s ready.", self._model)


class ApiProvider(EmbeddingProvider):
    """Calls any OpenAI-compatible /v1/embeddings endpoint.

    Works with: OpenAI, Azure OpenAI, Ollama's OpenAI-compatible surface,
    Together AI, Mistral, and any other compatible API.
    """

    def __init__(self, config: EmbedConfig, client: httpx.Client | None = None) -> None:
        self._url = config.url
        self._model = config.model
        self._api_key = config.api_key
        self._client = client or httpx.Client(timeout=60.0)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self._url}/v1/embeddings",
            headers=self._headers(),
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI returns {"data": [{"embedding": [...], "index": N}, ...]}
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def model_id(self) -> str:
        return f"api/{self._model}"


def build_provider(config: EmbedConfig, http_client: httpx.Client | None = None) -> EmbeddingProvider:
    if config.provider == "none":
        return NullProvider()
    if config.provider == "ollama":
        return OllamaProvider(config, http_client)
    if config.provider == "api":
        return ApiProvider(config, http_client)
    raise ValueError(f"Unknown provider: {config.provider}")
