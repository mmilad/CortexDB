"""Configuration for optional text processors."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.schemas.processor import ProcessorProvider, SidecarStrategy

_PROVIDER_ENV = "CORTEXDB_PROCESSOR_PROVIDER"
_URL_ENV = "CORTEXDB_PROCESSOR_URL"
_STRATEGY_ENV = "CORTEXDB_PROCESSOR_STRATEGY"


@dataclass(frozen=True)
class ProcessorConfig:
    provider: ProcessorProvider = "none"
    url: str = "http://127.0.0.1:5010"
    strategy: SidecarStrategy = "safe"

    @classmethod
    def from_env(cls) -> "ProcessorConfig":
        provider = os.environ.get(_PROVIDER_ENV, "none").lower()
        if provider not in ("none", "sidecar", "local"):
            provider = "none"

        strategy = os.environ.get(_STRATEGY_ENV, "safe").lower()
        if strategy not in ("safe", "semantic", "extractive"):
            strategy = "safe"

        return cls(
            provider=provider,  # type: ignore[arg-type]
            url=os.environ.get(_URL_ENV, "http://127.0.0.1:5010").rstrip("/"),
            strategy=strategy,  # type: ignore[arg-type]
        )
