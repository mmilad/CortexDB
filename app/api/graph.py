"""Graph traversal endpoint for CortexDB relationship graph."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.relationship import GraphExploreResponse
from app.services.graph import explore_graph
from app.store import SqliteStore, get_store

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get(
    "/explore",
    response_model=GraphExploreResponse,
    summary="BFS graph traversal from a starting node",
    description=(
        "Walk the relationship graph starting from a given dataset or tool key. "
        "start format: 'dataset:tech_knowledge' or 'tool:log_search'. "
        "depth controls max hops (1–5). Returns nodes and edges reachable within that depth."
    ),
)
def get_graph_explore(
    store: Annotated[SqliteStore, Depends(get_store)],
    start: str = Query(
        ...,
        description="Starting node: 'dataset:<key>' or 'tool:<key>'.",
        examples={"dataset": {"value": "dataset:tech_knowledge"}},
    ),
    depth: int = Query(
        default=2,
        ge=1,
        le=5,
        description="Maximum traversal depth (hops). Capped at 5.",
    ),
) -> GraphExploreResponse:
    datasets = store.list_datasets()
    tools = store.list_tools()
    edges = store.adjacency()
    return explore_graph(start, depth, datasets, tools, edges)
