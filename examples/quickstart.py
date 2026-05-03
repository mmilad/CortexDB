"""CortexDB quickstart — end-to-end example.

Demonstrates the full LLM-agent workflow against a running CortexDB server:

  1. Register two datasets and one tool with rich LLM metadata.
  2. Define a typed relationship between the datasets.
  3. Ingest raw text items into each dataset (vectorized by CortexDB).
  4. Run vector search, keyword search, and hybrid search.
  5. Call the LLM context endpoints to see what an agent would receive.
  6. Explore the relationship graph via BFS.
  7. Exercise the MCP JSON-RPC endpoint.

Requirements
------------
* CortexDB running locally:
    uvicorn app.main:app --reload

* Ollama with nomic-embed-text (default), OR set env vars for an
  OpenAI-compatible provider:
    CORTEXDB_EMBED_PROVIDER=api
    CORTEXDB_EMBED_URL=https://api.openai.com
    CORTEXDB_EMBED_MODEL=text-embedding-3-small
    CORTEXDB_EMBED_API_KEY=sk-...

Run
---
    python examples/quickstart.py [--base-url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DEFAULT_BASE = "http://127.0.0.1:8000"


def _check(r: httpx.Response, label: str) -> dict:
    if r.status_code not in (200, 201):
        print(f"  ✗ {label}: HTTP {r.status_code} — {r.text[:200]}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {label}")
    return r.json()


def _mcp(client: httpx.Client, base: str, method: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    r = client.post(f"{base}/mcp", json=body)
    assert r.status_code == 200, f"MCP {method} failed: {r.text}"
    return r.json()


def main(base: str) -> None:
    client = httpx.Client(base_url=base, timeout=60.0)

    # ------------------------------------------------------------------ #
    print("\n── 1. Health check ──")
    _check(client.get("/health"), "GET /health")

    # ------------------------------------------------------------------ #
    print("\n── 2. Register datasets ──")

    _check(client.post("/datasets", json={
        "dataset_key": "tech_knowledge",
        "display_name": "Technical Knowledge Base",
        "schema_version": "v1",
        "semantic_description": "Engineering documentation, guides, and resolved issues.",
        "usage_guidance": "Query when the user asks about technical topics, error messages, or best practices.",
        "llm_summary": "Internal tech KB — use for debugging help and engineering Q&A.",
        "retrieval_capabilities": ["vector", "keyword"],
        "content_kind": "documents",
        "entity_types": ["Article", "Guide", "Resolved Issue"],
        "access_patterns": ["semantic_search", "by_entity_id"],
        "filterable_fields": ["category", "severity"],
        "field_descriptions": [
            {"field": "category", "description": "Topic area", "example_values": ["networking", "database", "auth"]},
            {"field": "severity", "description": "Impact level", "example_values": ["low", "medium", "high"]},
        ],
        "query_examples": [
            {
                "label": "semantic_search",
                "description": "Find articles by meaning.",
                "example_request": {"query": "how to fix connection timeout", "top_k": 5},
            },
            {
                "label": "by_category",
                "description": "Filter by topic area.",
                "example_request": {
                    "query": "authentication failure",
                    "metadata_filters": {"category": "auth"},
                    "top_k": 10,
                },
            },
        ],
    }), "POST /datasets tech_knowledge")

    _check(client.post("/datasets", json={
        "dataset_key": "known_issues",
        "display_name": "Known Issues",
        "schema_version": "v1",
        "semantic_description": "Open and resolved incidents with symptoms and workarounds.",
        "usage_guidance": "Query when the user describes a problem that might already be tracked.",
        "llm_summary": "Incident tracker — use when the user reports a bug or outage.",
        "retrieval_capabilities": ["vector", "keyword"],
        "content_kind": "events",
        "entity_types": ["Incident", "Bug"],
        "filterable_fields": ["status", "severity"],
        "query_examples": [
            {
                "label": "open_by_severity",
                "description": "Find open high-severity incidents.",
                "example_request": {
                    "query": "database connection failure",
                    "metadata_filters": {"status": "open", "severity": "high"},
                    "top_k": 5,
                },
            },
        ],
    }), "POST /datasets known_issues")

    # ------------------------------------------------------------------ #
    print("\n── 3. Register a tool ──")

    _check(client.post("/tools", json={
        "tool_key": "log_search",
        "name": "Log Search",
        "description": "Full-text search over application logs.",
        "llm_summary": "Use to search application logs when diagnosing runtime errors.",
        "capability_tags": ["search", "logs", "diagnostics"],
        "safety_scope": "read-only",
        "input_schema_ref": json.dumps({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "start_time": {"type": "string", "format": "date-time"},
                "end_time": {"type": "string", "format": "date-time"},
                "service": {"type": "string"},
            },
            "required": ["query"],
        }),
        "status": "active",
    }), "POST /tools log_search")

    # ------------------------------------------------------------------ #
    print("\n── 4. Define a relationship ──")

    _check(client.post("/relationships", json={
        "source_type": "dataset",
        "source_key": "known_issues",
        "target_type": "dataset",
        "target_key": "tech_knowledge",
        "edge_type": "related",
        "join_fields": [],
        "description": "Incidents often reference KB articles as workarounds.",
    }), "POST /relationships known_issues → tech_knowledge")

    # ------------------------------------------------------------------ #
    print("\n── 5. Ingest items ──")

    tech_items = [
        {"raw_text": "Connection timeout errors often indicate misconfigured TCP keepalive settings. "
                     "Increase net.ipv4.tcp_keepalive_time to prevent idle connections from dropping.",
         "metadata": {"category": "networking", "severity": "medium"}},
        {"raw_text": "Database connection pool exhaustion: increase max_connections in postgresql.conf "
                     "and ensure all connections are properly released after use.",
         "metadata": {"category": "database", "severity": "high"}},
        {"raw_text": "JWT token expiration causes 401 Unauthorized. Implement token refresh logic "
                     "and validate exp claim on every request.",
         "metadata": {"category": "auth", "severity": "medium"}},
        {"raw_text": "Kubernetes pod OOMKilled: container exceeded memory limits. "
                     "Set appropriate resource requests and limits in the pod spec.",
         "metadata": {"category": "infrastructure", "severity": "high"}},
        {"raw_text": "Slow queries in PostgreSQL can be diagnosed with pg_stat_statements. "
                     "Enable it in shared_preload_libraries and analyze using EXPLAIN ANALYZE.",
         "metadata": {"category": "database", "severity": "low"}},
    ]
    r = client.post("/datasets/tech_knowledge/ingest", json={"items": tech_items})
    if r.status_code == 503:
        print("  ⚠ Embedding disabled — skipping vector ingest. Keyword search still works.")
        EMBEDDING_AVAILABLE = False
    else:
        _check(r, "POST /datasets/tech_knowledge/ingest (5 items)")
        EMBEDDING_AVAILABLE = True

    issues_items = [
        {"raw_text": "INC-001: Database connection pool exhausted during peak traffic. "
                     "Workaround: restart app server. Status: open.",
         "metadata": {"status": "open", "severity": "high"}},
        {"raw_text": "INC-002: JWT tokens not refreshing after password change. "
                     "Root cause: cache not invalidated. Status: resolved.",
         "metadata": {"status": "resolved", "severity": "medium"}},
        {"raw_text": "INC-003: Memory leak in image processing service. "
                     "Pod restarts every 6 hours. Status: in-progress.",
         "metadata": {"status": "in-progress", "severity": "high"}},
    ]
    r = client.post("/datasets/known_issues/ingest", json={"items": issues_items})
    if r.status_code != 503:
        _check(r, "POST /datasets/known_issues/ingest (3 items)")

    # ------------------------------------------------------------------ #
    print("\n── 6. Search ──")

    if EMBEDDING_AVAILABLE:
        r = _check(
            client.post("/datasets/tech_knowledge/search", json={
                "query": "database is running slowly",
                "top_k": 3,
            }),
            "Vector search: 'database is running slowly'",
        )
        print(f"     Top hit: {r['hits'][0]['item']['raw_text'][:80]}…")
        print(f"     Score:   {r['hits'][0]['score']}")

        r = _check(
            client.post("/datasets/tech_knowledge/search", json={
                "query": "authentication token expired",
                "keyword_query": "JWT",
                "vector_weight": 0.6,
                "top_k": 3,
            }),
            "Hybrid search: 'authentication token' + keyword 'JWT'",
        )
        print(f"     Top hit: {r['hits'][0]['item']['raw_text'][:80]}…")
        print(f"     Mode:    {r['search_mode']}")

    r = _check(
        client.post("/datasets/known_issues/search", json={
            "query": "any",
            "keyword_query": "database",
            "vector_weight": 0.0,
            "top_k": 5,
        }),
        "Keyword search: 'database' (no embedding needed)",
    )
    hit_count = len(r["hits"])
    print(f"     Hits: {hit_count}, mode: {r['search_mode']}")
    if hit_count == 0 and not EMBEDDING_AVAILABLE:
        print("     (0 hits expected — items were not ingested because embedding is disabled)")

    # ------------------------------------------------------------------ #
    print("\n── 7. LLM context endpoints ──")

    index = _check(client.get("/context/index"), "GET /context/index")
    print(f"     Datasets: {len(index['datasets'])}, Tools: {len(index['tools'])}, "
          f"Relationships: {index['relationship_count']}")

    ds_ctx = _check(client.get("/context/dataset/tech_knowledge"), "GET /context/dataset/tech_knowledge")
    print(f"     Query examples: {len(ds_ctx['query_examples'])}")

    graph = _check(client.get("/context/graph"), "GET /context/graph")
    print(f"     Graph edges: {len(graph['edges'])}")

    # ------------------------------------------------------------------ #
    print("\n── 8. Graph BFS traversal ──")

    bfs = _check(
        client.get("/graph/explore?start=dataset:known_issues&depth=2"),
        "GET /graph/explore?start=dataset:known_issues&depth=2",
    )
    print(f"     Nodes reachable: {len(bfs['nodes'])}")
    for node in bfs["nodes"]:
        node_type = node.get("node_type") or node.get("type", "unknown")
        print(f"       {node_type}:{node['key']}")

    # ------------------------------------------------------------------ #
    print("\n── 9. MCP JSON-RPC ──")

    init = _mcp(client, base, "initialize")
    print(f"  ✓ MCP initialize: protocol={init['result']['protocolVersion']}")

    resources = _mcp(client, base, "resources/list")
    uris = [r["uri"] for r in resources["result"]["resources"]]
    print(f"  ✓ resources/list: {len(uris)} resources")
    for uri in uris:
        print(f"       {uri}")

    ctx_read = _mcp(client, base, "resources/read", {"uri": "cortexdb://context/index"})
    index_data = json.loads(ctx_read["result"]["contents"][0]["text"])
    print(f"  ✓ resources/read cortexdb://context/index: "
          f"{len(index_data['datasets'])} datasets, {len(index_data['tools'])} tools")

    tools_list = _mcp(client, base, "tools/list")
    print(f"  ✓ tools/list: {len(tools_list['result']['tools'])} tools")

    # ------------------------------------------------------------------ #
    print("\n── 10. Discovery ──")

    disc = _check(
        client.post("/datasets/discover", json={"intent": "find known incidents about database"}),
        "POST /datasets/discover",
    )
    candidates = disc.get("candidates", [])
    print(f"     Action: {disc.get('recommended_action')}")
    print(f"     Candidates: {[c['dataset']['dataset_key'] for c in candidates]}")

    # ------------------------------------------------------------------ #
    print("\n✓ Quickstart complete — CortexDB is working end-to-end.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CortexDB quickstart")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="CortexDB base URL")
    args = parser.parse_args()
    main(args.base_url)
