from fastapi import APIRouter

from app.api import capabilities, context, datasets, graph, health, memory, relationships, tools
from app.mcp import server as mcp_server

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(capabilities.router)
api_router.include_router(datasets.router)
api_router.include_router(tools.router)
api_router.include_router(relationships.router)
api_router.include_router(memory.router)
api_router.include_router(context.router)
api_router.include_router(graph.router)
api_router.include_router(mcp_server.router)
