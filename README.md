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
| Embedding (nomic-embed-text / Ollama auto-start) | ✅ |
| Raw text ingest (`/datasets/{key}/ingest`) | ✅ |
| Vector + metadata search (`/datasets/{key}/search`) | ✅ |
| OpenAI-compatible API embedding provider | ✅ |
| sqlite-vec ANN index | Planned |
| Postgres + pgvector backend | Planned |
| Re-embedding jobs (raw text preserved) | Planned |
| Tenant / namespace isolation | Planned |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Storage: `cortexdb.sqlite` in the working directory.
Override: `CORTEXDB_DB_PATH=/path/to/file.sqlite`

Embedding defaults to **nomic-embed-text via Ollama**. On startup CortexDB
checks if Ollama is running; if not, it starts `ollama serve` and pulls the
model automatically. To use a different provider:

```bash
# Any OpenAI-compatible embedding API
CORTEXDB_EMBED_PROVIDER=api \
CORTEXDB_EMBED_URL=https://api.openai.com \
CORTEXDB_EMBED_MODEL=text-embedding-3-small \
CORTEXDB_EMBED_API_KEY=sk-... \
uvicorn app.main:app --reload

# Disable embedding entirely
CORTEXDB_EMBED_PROVIDER=none uvicorn app.main:app --reload
```

Callers **never send vectors** — only raw text. CortexDB handles vectorization.
Raw text is always stored, enabling re-embedding when models change.

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
