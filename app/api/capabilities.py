from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.store import SqliteStore, get_store

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict[str, Any]:
    datasets = store.list_datasets()
    tools = store.list_tools()
    rel_count = len(store.adjacency())
    return {
        "service": "cortexdb",
        "version": "0.2.0",
        "llm_inside": False,
        "resources": {
            "datasets": list(datasets.keys()),
            "tools": list(tools.keys()),
            "relationship_count": rel_count,
        },
        "docs": {
            "swagger_ui": "/docs",
            "openapi_json": "/openapi.json",
        },
        "llm_context_endpoints": {
            "index": "/context/index",
            "dataset_context": "/context/dataset/{key}",
            "tool_context": "/context/tool/{key}",
            "graph": "/context/graph",
        },
        "graph_endpoints": {
            "explore": "/graph/explore",
        },
        "mcp_endpoint": "/mcp",
    }
