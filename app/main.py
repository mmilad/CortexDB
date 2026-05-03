from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.embed.service import get_embedding_service
from app.store import close_store, get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Storage
    get_store()

    # Embedding service — starts Ollama if needed, pulls model, warms up
    embed_svc = get_embedding_service()
    embed_svc.startup()

    yield

    close_store()
    embed_svc.shutdown()


app = FastAPI(
    title="CortexDB API",
    version="0.3.0",
    description=(
        "CortexDB — LLM-native memory and retrieval layer. "
        "No generative LLM logic inside. "
        "Callers supply raw text; CortexDB handles vectorization, storage, "
        "retrieval, relationship graph, and LLM-optimised context interfaces."
    ),
    lifespan=lifespan,
)
app.include_router(api_router)
