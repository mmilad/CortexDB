# CortexDB — LLM-Native Database Strategy

## Executive Summary

CortexDB is a **memory and retrieval layer** for agentic/LLM systems. Its core design principle — *"LLM outside, memory intelligence inside"* — is sound. This document identifies the gaps between the current implementation and a genuinely LLM-native database, then describes an incremental build strategy that solves the five recurring LLM context problems, provides meaningful relationship management, and exposes a dynamic, low-token-cost MCP interface.

---

## 1. The Five Core LLM Context Problems CortexDB Must Solve

### Problem 1 — "What can I query?"
An LLM starting a task has no idea what data exists. It needs a compact, machine-readable index it can load in a single call at minimal token cost. The current `/capabilities` endpoint only returns a list of keys — not enough for the LLM to decide what to do.

### Problem 2 — "How do things relate?"
`relationship_hints` are free-text strings. An LLM cannot traverse them. There is no graph, no join path, no typed edge. When an LLM wants to query across two datasets (e.g. `known_issues` → `tech_knowledge`), it has no reliable guide.

### Problem 3 — "How do I write a good query?"
Datasets declare `filterable_fields` but do not show example queries, valid filter values, or query patterns. The LLM must guess, which causes hallucinated queries and wasted tokens on retries.

### Problem 4 — "How do I discover new data spaces?"
As datasets accumulate, the MCP or API surface must evolve. Right now, adding a new dataset doesn't update any discoverable capability in a structured way. A dynamic MCP that regenerates from the registry would solve this without code changes per dataset.

### Problem 5 — "How much context am I consuming?"
LLMs operate under tight token budgets. A single verbose registry dump of 50 datasets would consume thousands of tokens. CortexDB needs tiered context endpoints: a minimal index (names + one-line descriptions), then per-dataset deep context on demand.

---

## 2. Foundational Issues to Fix First

### 2.1 No Persistence
The current in-memory `RegistryState` loses all data on restart. Until there is at minimum an embedded database (SQLite), none of the advanced features can be built reliably.

**Fix:** Replace in-memory dicts with a SQLite-backed store using standard `sqlite3`. The store maintains the same Python interface so nothing above the state layer changes.

### 2.2 Relationships Are Documentation Only
`relationship_hints: list[str]` on both `DatasetRecord` and `ToolRecord` are informal strings. They work for humans reading JSON but cannot be traversed programmatically.

**Fix:** Introduce a first-class `RelationshipRecord` entity:
```
source_type:  "dataset" | "tool"
source_key:   str
target_type:  "dataset" | "tool"
target_key:   str
edge_type:    "joins_on" | "feeds_into" | "shared_entity" | "produces" | "consumes" | "related"
join_fields:  list[str]   # optional field names used to join
description:  str         # one sentence for the LLM
```

Relationships live in their own CRUD API and are stored in their own table. The graph traversal service reads edges from there and can follow them by hop depth.

### 2.3 Schemas Lack LLM-Usable Guidance
`DatasetRecord` has `semantic_description` and `usage_guidance`, but no:
- **`llm_summary`** — 1–2 sentence plain-English answer to "what is this for?"
- **`query_examples`** — concrete GET/filter/vector query payloads an LLM can adapt
- **`field_descriptions`** — per-field glossary so an LLM knows what `severity` means in context
- **`access_patterns`** — labeled patterns: `"by_time_range"`, `"by_entity_id"`, `"semantic_search"`, etc.

These additions do not require LLM inference inside CortexDB. They are authored by whoever registers the dataset and stored as structured metadata.

---

## 3. Build Layers (Ordered by Dependency)

### Layer 0 — Persistence (SQLite)
Replace `RegistryState` dicts with a single SQLite file. Tables:
- `datasets` — JSON blob keyed by `dataset_key`
- `tools` — JSON blob keyed by `tool_key`
- `relationships` — typed edges (new)
- `contexts` — pre-computed LLM context snapshots (new, optional cache)

This is zero-dependency (Python stdlib `sqlite3`). The API layer sees the same interface.

### Layer 1 — Rich Schemas
Extend `DatasetRecord` and `ToolRecord` with the LLM-guidance fields described in §2.3. All fields are optional so existing records remain valid.

### Layer 2 — Relationship Graph API
`/relationships` CRUD + `/graph/explore` traversal.

`GET /graph/explore?start=dataset:tech_knowledge&depth=2` returns:
```json
{
  "nodes": [{"key": "tech_knowledge", "type": "dataset"}, ...],
  "edges": [{"source": "tech_knowledge", "target": "known_issues", "edge_type": "related", ...}]
}
```

This is a lightweight adjacency-list traversal — no graph database required.

### Layer 3 — LLM Context Endpoints
Tiered endpoints designed specifically for low token usage:

| Endpoint | Purpose | Approx tokens |
|---|---|---|
| `GET /context/index` | One-line name + summary per dataset/tool | ~50 per item |
| `GET /context/dataset/{key}` | Full structured context for one dataset | ~300 per item |
| `GET /context/graph` | Relationship map (keys + edge types only) | ~30 per edge |
| `GET /context/mcp` | Full MCP-compatible capability descriptors | varies |

The LLM should first call `GET /context/index`, identify relevant items, then call `GET /context/dataset/{key}` only for those. This keeps initial orientation under 1,000 tokens for most deployments.

### Layer 4 — Dynamic MCP Server
An MCP server module (`app/mcp/`) that:
- Exposes one MCP `resource` per registered dataset (dynamically, from DB)
- Exposes one MCP `tool` per registered tool entry
- Exposes a `cortexdb://graph` resource for relationship traversal
- Exposes a `cortexdb://context/index` resource for the minimal index

The MCP server regenerates its exposed resources and tools from the SQLite registry on each call (or on a short TTL cache). Adding a new dataset via `POST /datasets` immediately makes it discoverable through MCP with no code change.

The MCP server is a **separate FastAPI/ASGI mount** (or standalone process) that talks to the same SQLite file. This keeps the REST API and the MCP server independently deployable.

---

## 4. Dynamic MCP — Core Design

### 4.1 Why MCP is the Right Interface
MCP (Model Context Protocol) is a structured way for an LLM to discover and call capabilities. For CortexDB the fit is natural:
- Each dataset becomes an MCP **resource** (readable structured data).
- Each tool in the ToolRegistry becomes an MCP **tool** (callable function with schema).
- The relationship graph becomes an MCP resource.
- The context index becomes an MCP resource.

The LLM can issue `resources/list` and `tools/list` to MCP and immediately know what exists, how to use it, and how items relate — all without a custom prompt.

### 4.2 Dynamic Generation Pattern

```
POST /datasets  ─► SQLite upsert
                      │
                      ▼
              MCP resources/list
                 (reads SQLite)
                      │
              returns new resource
              with URI, name, description
              and mimeType application/json
```

No code changes. No restarts (if using live DB reads). The MCP resource description comes from `llm_summary` + `query_examples`.

### 4.3 MCP Resource Schema per Dataset

```json
{
  "uri": "cortexdb://datasets/tech_knowledge",
  "name": "Tech Knowledge",
  "description": "<llm_summary> | Supports: vector, keyword | Related: known_issues",
  "mimeType": "application/json",
  "metadata": {
    "filterable_fields": ["component", "severity"],
    "retrieval_capabilities": ["vector", "keyword"],
    "query_examples": [...],
    "relationships": [...]
  }
}
```

The `description` field is crafted to be informative within ~100 tokens so `resources/list` remains cheap.

### 4.4 MCP Tool Schema per ToolRecord

```json
{
  "name": "log_search",
  "description": "<description> | Safe scope: <safety_scope>",
  "inputSchema": { "$ref": "<input_schema_ref resolved inline>" }
}
```

If `input_schema_ref` points to a URL or path, CortexDB resolves and inlines it. If not provided, a minimal `{"type": "object"}` schema is used.

### 4.5 MCP Growth Model
When a new dataset or tool is registered:
1. Developer calls `POST /datasets` or `POST /tools` with full metadata including `llm_summary` and `query_examples`.
2. MCP server returns it in the next `resources/list` / `tools/list` call.
3. Optionally, developer calls `POST /relationships` to declare edges from this new dataset to existing ones.
4. `GET /context/graph` updates automatically.

No MCP code changes. No server restarts. The "MCP grows as the registry grows" goal is met.

---

## 5. Graph-RAG Integration Path

CortexDB is not a GraphRAG engine — but it can be the **data substrate** one is built on. The path:

1. Datasets represent entity types (people, issues, documents, events).
2. Relationships describe typed edges between those entity types.
3. `GET /graph/explore` traverses the registry graph to produce a context window.
4. An external GraphRAG orchestrator uses this traversal to decide which datasets to query and in what order.
5. Results from each dataset query are composed into a graph-structured context before being sent to the LLM.

CortexDB's role: know the shape of the graph, expose traversal, return data per node. The LLM reasoning over the composed graph context remains external.

### Entity Resolution Hook
Add an `entity_types: list[str]` field to `DatasetRecord`. When the traversal service walks edges, it returns which entity types are found at each node. This lets an external orchestrator build a typed graph even without a triple store.

---

## 6. What the LLM Sees — Token Budget Analysis

### Minimal orientation (cold start)

```
GET /context/index
→ {"datasets": [
     {"key": "tech_knowledge", "summary": "Technical KB for engineering Q&A", "capabilities": ["vector", "keyword"]},
     {"key": "known_issues", "summary": "Open and resolved incidents", "capabilities": ["filter_only"]},
     ...
   ],
   "tools": [
     {"key": "log_search", "summary": "Search logs by metadata and vector hints"},
     ...
   ],
   "relationship_count": 4
  }
```

With 10 datasets and 5 tools this response is roughly **400–600 tokens** — affordable as a system context append.

### Focused context (after picking relevant items)

```
GET /context/dataset/tech_knowledge
→ {full record with llm_summary, filterable_fields, query_examples, related datasets}
```

This is **200–400 tokens** per dataset. An LLM picking 2–3 relevant datasets spends under 1,500 tokens total for orientation.

### Graph traversal (relationship understanding)

```
GET /context/graph
→ {"edges": [
     {"from": "tech_knowledge", "to": "known_issues", "edge_type": "related", "description": "Issues reference KB articles"},
     ...
   ]}
```

This is roughly **60 tokens per edge**. A graph with 20 edges is ~1,200 tokens.

---

## 7. Implementation Priorities

The implementation sequence follows dependency order. Each step is independently useful without requiring later steps.

### Step 1 — SQLite Persistence
- Replace `RegistryState` with `SqliteStore` (stdlib `sqlite3`).
- Tables: `datasets`, `tools`, `relationships`.
- No API changes.

### Step 2 — Enrich Schemas
- Add `llm_summary`, `query_examples`, `field_descriptions`, `access_patterns`, `entity_types` to `DatasetRecord`.
- Add `llm_summary`, `query_examples` to `ToolRecord`.
- All fields optional; backward compatible.

### Step 3 — Relationships API
- `RelationshipRecord` schema.
- `POST /relationships`, `GET /relationships`, `GET /relationships/{source_key}`, `DELETE /relationships/{id}`.
- Stored in SQLite.

### Step 4 — Context Endpoints
- `GET /context/index` — minimal token dump.
- `GET /context/dataset/{key}` — full context for one dataset.
- `GET /context/tool/{key}` — full context for one tool.
- `GET /context/graph` — relationship map.

### Step 5 — Graph Traversal Service
- `GET /graph/explore` with `start` and `depth` params.
- Adjacency-list BFS over the `relationships` table.
- Returns nodes + edges JSON.

### Step 6 — MCP Server Module
- `app/mcp/server.py` — MCP-over-HTTP server (or stdio transport).
- Dynamic resource generation from SQLite registry.
- Dynamic tool generation from ToolRegistry.
- Mounted at `/mcp` or as standalone process.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM hallucinates queries because `query_examples` are absent | Make `query_examples` a required field on `POST /datasets` (or warn loudly when absent). |
| `relationship_hints` strings coexist with `RelationshipRecord` and diverge | Deprecate `relationship_hints` once the relationship table is populated; keep backward compat during transition. |
| MCP resource list grows too large for `resources/list` to be usable | Add pagination + filtering to `resources/list`; also expose `GET /context/index` as a cheaper alternative. |
| SQLite write contention under concurrent ingestion | Use WAL mode (`PRAGMA journal_mode=WAL`); migrate to Postgres when needed. |
| Dataset metadata drifts from actual table schema | Add a `last_validated_at` field; expose a `/datasets/{key}/validate` hook. |
| No embedding versioning | Store `embedding_model_version` on every item; add re-embed jobs in Step 7+. |

---

## 9. Non-Goals (Remain Out of Scope)

- No LLM inference inside CortexDB. All reasoning stays external.
- No triple store or SPARQL. Typed edges on an adjacency list is sufficient for the GraphRAG substrate role.
- No complex ETL. Ingestion normalization stays minimal.
- No multi-tenant RBAC in the initial passes. Tenant fields are reserved in schema but not enforced until explicitly needed.

---

## 10. Summary Checklist

| Feature | Status | Priority |
|---|---|---|
| SQLite persistence | To build | Critical |
| `RelationshipRecord` + `/relationships` API | To build | Critical |
| `llm_summary`, `query_examples` on schemas | To build | High |
| `GET /context/index` | To build | High |
| `GET /context/dataset/{key}` | To build | High |
| `GET /context/graph` | To build | High |
| `GET /graph/explore` (BFS traversal) | To build | Medium |
| Dynamic MCP server module | To build | Medium |
| Hybrid retrieval (vector + keyword + filter) | Future | Medium |
| Re-embedding jobs | Future | Low |
| Tenant/namespace isolation | Future | Low |
| Scoring profiles in DB | Future | Low |
