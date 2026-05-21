from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app import __version__
from app.embed.service import EmbeddingService, get_embedding_service
from app.llm import LLMService, get_llm_service
from app.processors import ProcessorService, get_processor_service
from app.store import SqliteStore, get_store

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(
    store: Annotated[SqliteStore, Depends(get_store)],
    embed_svc: Annotated[EmbeddingService, Depends(get_embedding_service)],
    llm_svc: Annotated[LLMService, Depends(get_llm_service)],
    processor_svc: Annotated[ProcessorService, Depends(get_processor_service)],
) -> dict[str, Any]:
    datasets = store.list_datasets()
    tools = store.list_tools()
    rel_count = len(store.adjacency())
    return {
        "service": "cortexdb",
        "version": __version__,
        "llm_inside": False,
        "llm_provider": {
            "enabled": llm_svc.is_enabled(),
            "provider": llm_svc.provider,
            "model_id": llm_svc.model if llm_svc.is_enabled() else None,
            "url": llm_svc.url if llm_svc.is_enabled() else None,
        },
        "embedding": {
            "enabled": embed_svc.is_enabled(),
            "model_id": embed_svc.model_id if embed_svc.is_enabled() else None,
        },
        "processor": {
            "enabled": processor_svc.is_enabled(),
            "provider": processor_svc.provider,
            "url": processor_svc.url if processor_svc.provider == "sidecar" else None,
            "endpoints": {
                "health": "/processor/health",
                "process_text": "/processor/process/text",
                "analyze_ingest": "/processor/analyze/ingest",
            },
        },
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
        "memory_endpoints": {
            "ingest": "/datasets/{key}/ingest",
            "search": "/datasets/{key}/search (vector | keyword | hybrid)",
            "re_embed": "/datasets/{key}/re-embed",
            "items": "/datasets/{key}/items",
            "item": "/datasets/{key}/items/{id}",
            "soft_delete_item": "DELETE /datasets/{key}/items/{id}",
            "hard_delete_item": "DELETE /datasets/{key}/items/{id}/hard",
        },
        "validation_endpoint": "/datasets/{key}/validate",
        "mcp_endpoint": "/mcp",
    }
