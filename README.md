# CortexDB

CortexDB is a **LLM-native memory and retrieval layer** for agentic systems.

## Core principles

- **No LLM logic inside CortexDB.** All reasoning stays external.
- CortexDB stores, indexes, filters, and scores data deterministically.
- Consumer applications provide raw text and perform reasoning externally.
- Every dataset and tool is self-describing in a format LLMs can consume efficiently.

## Feature status

| Feature | Status |
|---|---|
| Dataset registry (CRUD + discovery) | ✅ |
| Tool registry (CRUD) | ✅ |
| SQLite persistence (survives restarts) | ✅ |
| Typed relationship graph (`/relationships`) | ✅ |
| BFS graph traversal (`/graph/explore`) | ✅ |
| LLM context endpoints (`/context/*`) | ✅ |
| Dynamic MCP server (`/mcp` HTTP + stdio) | ✅ |
| Embedding (nomic-embed-text / Ollama auto-start) | ✅ |
| Raw text ingest (`/datasets/{key}/ingest`) | ✅ |
| Vector + metadata search (`/datasets/{key}/search`) | ✅ |
| BM25 keyword scoring (hybrid search) | ✅ |
| OpenAI-compatible API embedding provider | ✅ |
| sqlite-vec ANN index (optional, auto-detected) | ✅ |
| Relationship cascade delete | ✅ |
| MCP `input_schema_ref` URL resolution | ✅ |
| Re-embedding jobs (`/datasets/{key}/re-embed`) | ✅ |
| Dataset metadata validation (`/datasets/{key}/validate`) | ✅ |
| Postgres + pgvector backend | Planned |
| Tenant / namespace isolation | Planned |
| Scoring profiles stored in DB | Planned |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[vec]"     # includes sqlite-vec for ANN search
cortexdb-api --reload       # defaults to http://127.0.0.1:5000
```

Storage: `cortexdb.sqlite` in the working directory.
Override: `CORTEXDB_DB_PATH=/path/to/file.sqlite`
API port override: `CORTEXDB_API_PORT=5001 cortexdb-api --reload`
Editable local defaults: copy `.env.example` or export the `CORTEXDB_API_*` variables in your shell before starting the server.

Embedding defaults to **nomic-embed-text via Ollama**. On startup CortexDB
checks if Ollama is running; if not, it starts `ollama serve` and pulls the
model automatically. To use a different provider:

```bash
# Any OpenAI-compatible embedding API
CORTEXDB_EMBED_PROVIDER=api \
CORTEXDB_EMBED_URL=https://api.openai.com \
CORTEXDB_EMBED_MODEL=text-embedding-3-small \
CORTEXDB_EMBED_API_KEY=sk-... \
cortexdb-api --reload

# Disable embedding entirely
CORTEXDB_EMBED_PROVIDER=none cortexdb-api --reload
```

Callers **never send vectors** — only raw text. CortexDB handles vectorization.
Raw text is always stored, enabling re-embedding when models change.

API docs: `http://127.0.0.1:5000/docs`

End-to-end example: `python examples/quickstart.py`

The safe deterministic text processor runs in-process by default, so
`cortexdb-api` is enough for normal ingest and processor usage. Its HTTP helper
endpoints are mounted under `/processor/*`, for example
`POST /processor/process/text` and `POST /processor/analyze/ingest`. If you want
the processor isolated as a separate process, run `cortexdb-processor --reload`
and set `CORTEXDB_PROCESSOR_CLIENT_PROVIDER=sidecar`.

For deterministic chunking of text, Markdown files, and directories before
ingest, see the [Ingest Pipeline usage guide](./docs/USAGE.md#ingest-pipeline).

## Testing

Run tests with embedding disabled to avoid external model/provider dependencies:

```bash
export CORTEXDB_EMBED_PROVIDER=none
pytest
```

For a realistic API flow using `FastAPI TestClient` and an isolated temporary
SQLite database:

```bash
pytest tests/test_simulated_usage.py -q
```

### sqlite-vec ANN index

When `sqlite-vec` is installed (`pip install -e ".[vec]"`), CortexDB
automatically creates a `vec0` ANN index per dataset on first ingest and uses
it for all vector searches (~19 ms at 20 000 rows / 768 dim). Without it,
the store falls back to an in-process Python cosine scan (suitable up to ~20 k
rows). The `embedding` TEXT column is always the source of truth — the vec0
table can be rebuilt at any time via `POST /datasets/{key}/re-embed`.

## LLM Agent Orientation Pattern

An LLM agent starting a task should:

```
GET /context/index          → what datasets and tools exist? (~50 tokens per item)
GET /context/dataset/{key}  → full query guidance for a specific dataset
GET /context/graph          → how datasets and tools relate to each other
GET /graph/explore?start=.. → BFS subgraph from a starting node
```

Or via MCP (HTTP):

```
resources/list                                    → all registered datasets and tools as MCP resources
resources/read cortexdb://context/index           → minimal orientation
resources/read cortexdb://datasets/{key}          → full dataset context
resources/read cortexdb://graph                   → relationship map
```

Or via MCP (stdio — for Claude Desktop, Cursor, Continue, etc.):

```bash
# Run the stdio MCP server directly
python -m app.mcp.stdio

# Or use the installed entry point
cortexdb-mcp
```

Example `claude_desktop_config.json` entry:
```json
{
  "mcpServers": {
    "cortexdb": {
      "command": "cortexdb-mcp",
      "env": {
        "CORTEXDB_DB_PATH": "/path/to/cortexdb.sqlite",
        "CORTEXDB_EMBED_PROVIDER": "none"
      }
    }
  }
}
```

Adding a new dataset via `POST /datasets` automatically updates MCP `resources/list`
and `GET /context/index` — no code changes or restarts needed.

## Testing

```bash
export CORTEXDB_EMBED_PROVIDER=none
pip install -e '.[dev]'
pytest
```

109 tests, no Ollama or running server required.

## Docs

- Agent operations guide: [`AGENTS.md`](./AGENTS.md) — **start here if you are an AI agent**
- Usage guide with curl examples: [`docs/USAGE.md`](./docs/USAGE.md)
- Architecture and design decisions: [`ARCHITECTURE_PLAN.md`](./ARCHITECTURE_PLAN.md)
- Historical strategy and design rationale: [`STRATEGY.md`](./STRATEGY.md)
