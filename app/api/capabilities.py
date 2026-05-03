from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.state import RegistryState, get_registry

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> dict[str, Any]:
    return {
        "service": "cortexdb",
        "llm_inside": False,
        "resources": {
            "datasets": list(reg.datasets.keys()),
            "tools": list(reg.tools.keys()),
        },
        "docs": {"swagger_ui": "/docs", "openapi_json": "/openapi.json"},
    }
