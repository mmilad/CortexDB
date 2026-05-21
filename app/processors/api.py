"""FastAPI app for the optional CortexDB processor sidecar."""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.processors.router import router

app = FastAPI(
    title="CortexDB Processor",
    description="Optional long-lived text processor sidecar for CortexDB.",
    version=__version__,
)
app.include_router(router)
