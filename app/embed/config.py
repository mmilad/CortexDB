"""Embedding configuration resolved from environment variables.

CORTEXDB_EMBED_PROVIDER   = ollama (default) | api | none
CORTEXDB_EMBED_MODEL      = nomic-embed-text (default)
CORTEXDB_EMBED_URL        = base URL for the provider
                            ollama default: http://localhost:11434
                            api default:    https://api.openai.com
CORTEXDB_EMBED_API_KEY    = API key (api provider only; optional)
CORTEXDB_OLLAMA_AUTOSTART = true (default) — start ollama serve on startup
                            when provider=ollama and Ollama is not reachable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["ollama", "api", "none"]

_DEFAULTS: dict[str, str] = {
    "CORTEXDB_EMBED_PROVIDER": "ollama",
    "CORTEXDB_EMBED_MODEL": "nomic-embed-text",
    "CORTEXDB_EMBED_URL_OLLAMA": "http://localhost:11434",
    "CORTEXDB_EMBED_URL_API": "https://api.openai.com",
    "CORTEXDB_OLLAMA_AUTOSTART": "true",
}


@dataclass(frozen=True)
class EmbedConfig:
    provider: Provider
    model: str
    url: str
    api_key: str | None
    ollama_autostart: bool

    @classmethod
    def from_env(cls) -> "EmbedConfig":
        provider_raw = os.environ.get("CORTEXDB_EMBED_PROVIDER", _DEFAULTS["CORTEXDB_EMBED_PROVIDER"]).lower()
        if provider_raw not in ("ollama", "api", "none"):
            raise ValueError(
                f"CORTEXDB_EMBED_PROVIDER must be 'ollama', 'api', or 'none'. Got: {provider_raw!r}"
            )
        provider: Provider = provider_raw  # type: ignore[assignment]

        model = os.environ.get("CORTEXDB_EMBED_MODEL", _DEFAULTS["CORTEXDB_EMBED_MODEL"])

        if provider == "ollama":
            url = os.environ.get("CORTEXDB_EMBED_URL", _DEFAULTS["CORTEXDB_EMBED_URL_OLLAMA"])
        elif provider == "api":
            url = os.environ.get("CORTEXDB_EMBED_URL", _DEFAULTS["CORTEXDB_EMBED_URL_API"])
        else:
            url = ""

        api_key = os.environ.get("CORTEXDB_EMBED_API_KEY") or None

        autostart_raw = os.environ.get("CORTEXDB_OLLAMA_AUTOSTART", _DEFAULTS["CORTEXDB_OLLAMA_AUTOSTART"])
        ollama_autostart = autostart_raw.lower() in ("1", "true", "yes")

        return cls(
            provider=provider,
            model=model,
            url=url.rstrip("/"),
            api_key=api_key,
            ollama_autostart=ollama_autostart,
        )
