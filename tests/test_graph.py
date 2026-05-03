"""Unit tests for BFS graph traversal."""
from __future__ import annotations

import pytest

from app.services.graph import explore_graph


def _make_edge(id_, src, tgt, edge_type="related"):
    return {
        "id": id_,
        "source_type": "dataset",
        "source_key": src,
        "target_type": "dataset",
        "target_key": tgt,
        "edge_type": edge_type,
        "join_fields": [],
        "description": "",
    }


DATASETS = {
    "a": {"display_name": "A", "llm_summary": "Node A", "entity_types": []},
    "b": {"display_name": "B", "llm_summary": "Node B", "entity_types": []},
    "c": {"display_name": "C", "llm_summary": "Node C", "entity_types": []},
    "d": {"display_name": "D", "llm_summary": "Node D", "entity_types": []},
}

EDGES = [
    _make_edge("e1", "a", "b"),
    _make_edge("e2", "b", "c"),
    _make_edge("e3", "c", "d"),
]


def test_depth_1():
    result = explore_graph("dataset:a", depth=1, datasets=DATASETS, tools={}, edges=EDGES)
    node_keys = {n.key for n in result.nodes}
    assert "a" in node_keys
    assert "b" in node_keys
    assert "c" not in node_keys


def test_depth_2():
    result = explore_graph("dataset:a", depth=2, datasets=DATASETS, tools={}, edges=EDGES)
    node_keys = {n.key for n in result.nodes}
    assert {"a", "b", "c"}.issubset(node_keys)
    assert "d" not in node_keys


def test_depth_3():
    result = explore_graph("dataset:a", depth=3, datasets=DATASETS, tools={}, edges=EDGES)
    node_keys = {n.key for n in result.nodes}
    assert {"a", "b", "c", "d"}.issubset(node_keys)


def test_unknown_start_returns_root_only():
    result = explore_graph("dataset:zzz", depth=2, datasets=DATASETS, tools={}, edges=EDGES)
    assert len(result.nodes) == 1
    assert result.nodes[0].key == "zzz"


def test_no_edges():
    result = explore_graph("dataset:a", depth=2, datasets=DATASETS, tools={}, edges=[])
    assert len(result.nodes) == 1


def test_cycle_does_not_loop():
    cycle_edges = [
        _make_edge("c1", "x", "y"),
        _make_edge("c2", "y", "x"),
    ]
    datasets = {"x": {}, "y": {}}
    result = explore_graph("dataset:x", depth=5, datasets=datasets, tools={}, edges=cycle_edges)
    node_keys = {n.key for n in result.nodes}
    assert node_keys == {"x", "y"}


def test_start_without_type_prefix_defaults_to_dataset():
    result = explore_graph("a", depth=1, datasets=DATASETS, tools={}, edges=EDGES)
    assert result.root == "a"
    assert any(n.key == "b" for n in result.nodes)


def test_tool_node():
    tools = {"t1": {"name": "Tool One", "llm_summary": "Does stuff"}}
    edges = [{"id": "te1", "source_type": "dataset", "source_key": "a",
              "target_type": "tool", "target_key": "t1",
              "edge_type": "produces", "join_fields": [], "description": ""}]
    result = explore_graph("dataset:a", depth=1, datasets=DATASETS, tools=tools, edges=edges)
    tool_nodes = [n for n in result.nodes if n.node_type == "tool"]
    assert len(tool_nodes) == 1
    assert tool_nodes[0].key == "t1"
    assert tool_nodes[0].display_name == "Tool One"
