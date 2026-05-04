from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.router import api_router
from app.embed.service import get_embedding_service
from app.store import close_store, get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()

    embed_svc = get_embedding_service()
    await embed_svc.startup()

    yield

    close_store()
    await embed_svc.shutdown()


app = FastAPI(
    title="CortexDB API",
    version=__version__,
    description=(
        "CortexDB — LLM-native memory and retrieval layer. "
        "No generative LLM logic inside. "
        "Callers supply raw text; CortexDB handles vectorization, storage, "
        "retrieval, relationship graph, and LLM-optimised context interfaces."
    ),
    lifespan=lifespan,
)
app.include_router(api_router)
