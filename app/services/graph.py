"""BFS graph traversal over the CortexDB relationship table.

No graph database required — walks the adjacency list stored in SQLite.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from app.schemas.relationship import GraphEdge, GraphExploreResponse, GraphNode


def _parse_start(start: str) -> tuple[str, str]:
    """Parse 'dataset:tech_knowledge' or 'tool:log_search' into (type, key)."""
    if ":" in start:
        node_type, key = start.split(":", 1)
        if node_type not in ("dataset", "tool"):
            raise ValueError(f"node_type must be 'dataset' or 'tool', got '{node_type}'")
        return node_type, key
    return "dataset", start


def explore_graph(
    start: str,
    depth: int,
    datasets: dict[str, dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> GraphExploreResponse:
    """BFS starting from `start` up to `depth` hops.

    Parameters
    ----------
    start:    "dataset:key" or "tool:key" (type prefix is optional; defaults to dataset).
    depth:    Maximum number of hops from the root node.
    datasets: All dataset records keyed by dataset_key.
    tools:    All tool records keyed by tool_key.
    edges:    All relationship rows from the store.
    """
    start_type, start_key = _parse_start(start)

    # Build adjacency: node_id → list of (neighbour_id, edge_dict)
    # node_id = f"{type}:{key}"
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for e in edges:
        src = f"{e['source_type']}:{e['source_key']}"
        tgt = f"{e['target_type']}:{e['target_key']}"
        adjacency.setdefault(src, []).append((tgt, e))
        adjacency.setdefault(tgt, []).append((src, e))  # undirected traversal

    root_id = f"{start_type}:{start_key}"
    visited: dict[str, int] = {root_id: 0}  # node_id → hop distance
    queue: deque[tuple[str, int]] = deque([(root_id, 0)])
    seen_edges: set[frozenset] = set()
    result_edges: list[GraphEdge] = []

    while queue:
        current, dist = queue.popleft()
        if dist >= depth:
            continue
        for neighbour, edge_data in adjacency.get(current, []):
            edge_key = frozenset([edge_data["id"]])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                result_edges.append(
                    GraphEdge(
                        source=edge_data["source_key"],
                        target=edge_data["target_key"],
                        edge_type=edge_data["edge_type"],
                        description=edge_data.get("description", ""),
                        join_fields=edge_data.get("join_fields", []),
                    )
                )
            if neighbour not in visited:
                visited[neighbour] = dist + 1
                queue.append((neighbour, dist + 1))

    # Build node list from visited set
    result_nodes: list[GraphNode] = []
    for node_id in visited:
        ntype, nkey = node_id.split(":", 1)
        if ntype == "dataset":
            rec = datasets.get(nkey, {})
            result_nodes.append(
                GraphNode(
                    key=nkey,
                    node_type="dataset",
                    display_name=rec.get("display_name"),
                    llm_summary=rec.get("llm_summary"),
                    entity_types=rec.get("entity_types", []),
                )
            )
        else:
            rec = tools.get(nkey, {})
            result_nodes.append(
                GraphNode(
                    key=nkey,
                    node_type="tool",
                    display_name=rec.get("name"),
                    llm_summary=rec.get("llm_summary"),
                    entity_types=[],
                )
            )

    return GraphExploreResponse(
        root=start_key,
        depth=depth,
        nodes=result_nodes,
        edges=result_edges,
    )
