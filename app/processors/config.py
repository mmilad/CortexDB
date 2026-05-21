"""Configuration for optional text processors."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.schemas.processor import ProcessorProvider, SidecarStrategy

_PROVIDER_ENV = "CORTEXDB_PROCESSOR_CLIENT_PROVIDER"
_URL_ENV = "CORTEXDB_PROCESSOR_CLIENT_URL"
_STRATEGY_ENV = "CORTEXDB_PROCESSOR_CLIENT_STRATEGY"
_CLASSIFY_ENV = "CORTEXDB_PROCESSOR_CLIENT_CLASSIFY"
_KNOWN_THRESHOLD_ENV = "CORTEXDB_PROCESSOR_CLIENT_KNOWN_THRESHOLD"
_CANDIDATE_THRESHOLD_ENV = "CORTEXDB_PROCESSOR_CLIENT_CANDIDATE_THRESHOLD"
_FALLBACK_ENV = "CORTEXDB_PROCESSOR_CLIENT_FALLBACK"

_LEGACY_PROVIDER_ENV = "CORTEXDB_PROCESSOR_PROVIDER"
_LEGACY_URL_ENV = "CORTEXDB_PROCESSOR_URL"
_LEGACY_STRATEGY_ENV = "CORTEXDB_PROCESSOR_STRATEGY"
_LEGACY_CLASSIFY_ENV = "CORTEXDB_PROCESSOR_CLASSIFY"
_LEGACY_KNOWN_THRESHOLD_ENV = "CORTEXDB_PROCESSOR_KNOWN_THRESHOLD"
_LEGACY_CANDIDATE_THRESHOLD_ENV = "CORTEXDB_PROCESSOR_CANDIDATE_THRESHOLD"
_LEGACY_FALLBACK_ENV = "CORTEXDB_PROCESSOR_FALLBACK"


def _env(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary, os.environ.get(legacy, default))


@dataclass(frozen=True)
class ProcessorConfig:
    provider: ProcessorProvider = "local"
    url: str = "http://127.0.0.1:5010"
    strategy: SidecarStrategy = "safe"
    classify: bool = False
    known_match_threshold: float = 0.72
    candidate_threshold: float = 0.45
    graceful_fallback: bool = True

    @classmethod
    def from_env(cls) -> "ProcessorConfig":
        provider = _env(_PROVIDER_ENV, _LEGACY_PROVIDER_ENV, "local").lower()
        if provider not in ("none", "sidecar", "local"):
            provider = "none"

        strategy = _env(_STRATEGY_ENV, _LEGACY_STRATEGY_ENV, "safe").lower()
        if strategy not in ("safe", "semantic", "extractive"):
            strategy = "safe"

        def _float_env(primary: str, legacy: str, default: float) -> float:
            try:
                value = float(_env(primary, legacy, str(default)))
            except ValueError:
                return default
            return min(1.0, max(0.0, value))

        return cls(
            provider=provider,  # type: ignore[arg-type]
            url=_env(_URL_ENV, _LEGACY_URL_ENV, "http://127.0.0.1:5010").rstrip("/"),
            strategy=strategy,  # type: ignore[arg-type]
            classify=_env(_CLASSIFY_ENV, _LEGACY_CLASSIFY_ENV, "false").lower() in ("1", "true", "yes", "on"),
            known_match_threshold=_float_env(_KNOWN_THRESHOLD_ENV, _LEGACY_KNOWN_THRESHOLD_ENV, 0.72),
            candidate_threshold=_float_env(_CANDIDATE_THRESHOLD_ENV, _LEGACY_CANDIDATE_THRESHOLD_ENV, 0.45),
            graceful_fallback=_env(_FALLBACK_ENV, _LEGACY_FALLBACK_ENV, "true").lower()
            not in ("0", "false", "no", "off"),
        )
