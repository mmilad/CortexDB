"""Main-API client for optional processor providers."""

from __future__ import annotations

import logging

import httpx

from app.processors.config import ProcessorConfig
from app.processors.safe import process_text_safe
from app.schemas.processor import ProcessorRequest, ProcessorResponse

logger = logging.getLogger("cortexdb.processors.service")


class ProcessorService:
    def __init__(self) -> None:
        self._config = ProcessorConfig.from_env()

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def url(self) -> str:
        return self._config.url

    def is_enabled(self) -> bool:
        return self._config.provider != "none"

    async def process_text(self, request: ProcessorRequest) -> ProcessorResponse:
        if self._config.provider == "none":
            raise RuntimeError("Processor service is disabled.")

        if self._config.provider == "local":
            if request.strategy in ("semantic", "extractive"):
                raise NotImplementedError(f"processor strategy '{request.strategy}' is not implemented yet")
            return process_text_safe(request)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._config.url}/process/text",
                json=request.model_dump(),
            )
            response.raise_for_status()
        return ProcessorResponse.model_validate(response.json())


def get_processor_service() -> ProcessorService:
    return ProcessorService()
