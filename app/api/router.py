from fastapi import APIRouter

from app.api import capabilities, context, datasets, graph, health, ingest, ingest_rules, memory, relationships, sessions, tools
from app.mcp import server as mcp_server

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(capabilities.router)
api_router.include_router(datasets.router)
api_router.include_router(tools.router)
api_router.include_router(relationships.router)
api_router.include_router(memory.router)
api_router.include_router(ingest.router)
api_router.include_router(ingest_rules.router)
api_router.include_router(sessions.router)
api_router.include_router(context.router)
api_router.include_router(graph.router)
api_router.include_router(mcp_server.router)
