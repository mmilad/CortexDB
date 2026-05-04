"""Embedding service — singleton that manages provider lifecycle.

On startup (called from app lifespan):
  1. Read EmbedConfig from env.
  2. If provider=ollama and CORTEXDB_OLLAMA_AUTOSTART=true:
     a. Check if Ollama is reachable at the configured URL.
     b. If not reachable, start `ollama serve` as a managed subprocess.
     c. Wait for it to be ready (up to 30s with backoff).
     d. Ensure the configured model is pulled.
  3. Warm up with a single test embed to catch misconfiguration early.

On shutdown: close the async HTTP client and terminate the managed Ollama
subprocess if we started it.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time

import httpx

from app.embed.config import EmbedConfig
from app.embed.providers import EmbeddingProvider, OllamaProvider, build_provider

logger = logging.getLogger("cortexdb.embed.service")

_OLLAMA_READY_TIMEOUT = 30  # seconds
_OLLAMA_READY_POLL = 1.0    # seconds between health checks


class EmbeddingService:
    """Manages one EmbeddingProvider for the lifetime of the application."""

    def __init__(self) -> None:
        self._config: EmbedConfig | None = None
        self._provider: EmbeddingProvider | None = None
        self._ollama_proc: subprocess.Popen | None = None
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        self._config = EmbedConfig.from_env()
        logger.info("Embedding provider: %s / model: %s", self._config.provider, self._config.model)

        if self._config.provider == "none":
            from app.embed.providers import NullProvider
            self._provider = NullProvider()
            logger.info("Embedding disabled.")
            return

        self._http = httpx.AsyncClient(timeout=60.0)

        if self._config.provider == "ollama" and self._config.ollama_autostart:
            await self._ensure_ollama_running()

        self._provider = build_provider(self._config, self._http)

        if self._config.provider == "ollama":
            assert isinstance(self._provider, OllamaProvider)
            try:
                await self._provider.ensure_model_pulled()
            except Exception as exc:
                logger.warning("Could not pull model %s: %s", self._config.model, exc)

        try:
            await self._provider.embed_one("warmup")
            logger.info("Embedding service ready. model_id=%s", self._provider.model_id())
        except Exception as exc:
            logger.warning("Embedding warm-up failed: %s. Service will retry on first real call.", exc)

    async def shutdown(self) -> None:
        if self._ollama_proc is not None:
            logger.info("Stopping managed Ollama process (pid=%s)...", self._ollama_proc.pid)
            self._ollama_proc.terminate()
            try:
                self._ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ollama_proc.kill()
            self._ollama_proc = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._provider = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            raise RuntimeError("EmbeddingService.startup() has not been called.")
        return self._provider

    @property
    def model_id(self) -> str:
        return self.provider.model_id()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.provider.embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        return await self.provider.embed_one(text)

    def is_enabled(self) -> bool:
        return self._config is not None and self._config.provider != "none"

    # ------------------------------------------------------------------
    # Ollama auto-start
    # ------------------------------------------------------------------

    async def _ollama_reachable(self) -> bool:
        assert self._config is not None
        assert self._http is not None
        try:
            resp = await self._http.get(f"{self._config.url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def _ensure_ollama_running(self) -> None:
        assert self._config is not None

        if await self._ollama_reachable():
            logger.info("Ollama already running at %s.", self._config.url)
            return

        logger.info(
            "Ollama not reachable at %s. Starting 'ollama serve'...", self._config.url
        )
        try:
            self._ollama_proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning(
                "ollama binary not found. Install Ollama from https://ollama.com "
                "or set CORTEXDB_EMBED_PROVIDER=api to use an external API instead."
            )
            return

        deadline = time.monotonic() + _OLLAMA_READY_TIMEOUT
        while time.monotonic() < deadline:
            if await self._ollama_reachable():
                logger.info("Ollama started (pid=%s).", self._ollama_proc.pid)
                return
            await asyncio.sleep(_OLLAMA_READY_POLL)

        logger.warning(
            "Ollama did not become ready within %ds. "
            "Embedding calls may fail until it is available.",
            _OLLAMA_READY_TIMEOUT,
        )


# ------------------------------------------------------------------
# Singleton + FastAPI dependency
# ------------------------------------------------------------------

_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    return _service
