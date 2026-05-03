# CortexDB

CortexDB is a **LLM-native memory and retrieval layer** for agentic systems.

## Core principles

- **No LLM logic inside CortexDB.** All reasoning stays external.
- CortexDB stores, indexes, filters, and scores data deterministically.
- Consumer applications provide embeddings/intents and perform reasoning externally.
- Every dataset and tool is self-describing in a format LLMs can consume efficiently.

## What's in here

| Feature | Status |
|---|---|
| Dataset registry (CRUD + discovery) | ✅ |
| Tool registry (CRUD) | ✅ |
| SQLite persistence (survives restarts) | ✅ |
| Typed relationship graph (`/relationships`) | ✅ |
| BFS graph traversal (`/graph/explore`) | ✅ |
| LLM context endpoints (`/context/*`) | ✅ |
| Dynamic MCP server (`/mcp`) | ✅ |
| Vector / hybrid retrieval | Planned |
| Tenant / namespace isolation | Planned |
| Re-embedding jobs | Planned |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Storage: `cortexdb.sqlite` in the working directory.
Override: `CORTEXDB_DB_PATH=/path/to/file.sqlite uvicorn app.main:app --reload`

API docs: `http://127.0.0.1:8000/docs`

## LLM Agent Orientation Pattern

An LLM agent starting a task should:

```
GET /context/index          → what datasets and tools exist? (~50 tokens per item)
GET /context/dataset/{key}  → full query guidance for a specific dataset
GET /context/graph          → how datasets and tools relate to each other
GET /graph/explore?start=.. → BFS subgraph from a starting node
```

Or via MCP:

```
resources/list              → all registered datasets and tools as MCP resources
resources/read cortexdb://context/index   → minimal orientation
resources/read cortexdb://datasets/{key}  → full dataset context
resources/read cortexdb://graph           → relationship map
```

Adding a new dataset via `POST /datasets` automatically updates MCP `resources/list`
and `GET /context/index` — no code changes or restarts needed.

## Docs

- Strategy & design: [`STRATEGY.md`](./STRATEGY.md)
- Architecture plan: [`ARCHITECTURE_PLAN.md`](./ARCHITECTURE_PLAN.md)
- Usage guide: [`docs/USAGE.md`](./docs/USAGE.md)
