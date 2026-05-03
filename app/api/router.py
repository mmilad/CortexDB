from fastapi import APIRouter

from app.api import capabilities, datasets, health, tools

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(capabilities.router)
api_router.include_router(datasets.router)
api_router.include_router(tools.router)
