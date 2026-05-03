from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI(
    title="CortexDB API",
    version="0.1.0",
    description=(
        "CortexDB service API (no internal LLM logic). "
        "Provides deterministic memory, registry, and retrieval metadata interfaces."
    ),
)
app.include_router(api_router)
