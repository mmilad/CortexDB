from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.store import close_store, get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()  # initialise + run DDL on startup
    yield
    close_store()


app = FastAPI(
    title="CortexDB API",
    version="0.2.0",
    description=(
        "CortexDB service API (no internal LLM logic). "
        "Provides deterministic memory, registry, retrieval metadata, "
        "relationship graph, and LLM-optimised context interfaces."
    ),
    lifespan=lifespan,
)
app.include_router(api_router)
