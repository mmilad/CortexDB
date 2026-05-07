# CortexDB — AI Agent Operations Guide

This file is the authoritative reference for any AI agent (Cursor Cloud, Claude, GPT, etc.) working on this repository. Read it before touching any code.

---

## Project summary

CortexDB is a **LLM-native memory and retrieval layer** for agentic systems.

Design principle: **"LLM outside, memory intelligence inside."**

- Stores datasets, tools, relationships, and raw-text memory items in SQLite.
- Exposes a REST API (FastAPI) and a dynamic MCP server (HTTP + stdio).
- Never runs generative LLM inference internally.
- Callers send raw text; CortexDB handles vectorization via pluggable providers.

---

## Repository layout

```
app/
  main.py               FastAPI app entry point + lifespan hooks
  api/                  REST route handlers (one module per resource group)
    capabilities.py     Legacy /capabilities endpoint
    context.py          /context/* LLM-optimised context endpoints
    datasets.py         /datasets CRUD + /discover + /validate
    graph.py            /graph/explore BFS traversal
    health.py           /health liveness check
    memory.py           /datasets/{key}/ingest|search|re-embed|items CRUD
    relationships.py    /relationships CRUD
    router.py           Combines all routers + MCP mount
    tools.py            /tools CRUD
  context_builders.py   Pure functions: build_context_index, build_dataset_payload, etc.
  embed/
    config.py           EmbedConfig from env vars
    providers.py        Ollama + OpenAI-compatible HTTP providers
    service.py          EmbeddingService singleton
  mcp/
    server.py           MCP JSON-RPC over HTTP (mounted at /mcp)
    stdio.py            MCP stdio transport + cortexdb-mcp entry point
  schemas/              Pydantic models (DatasetRecord, ToolRecord, …)
  services/
    dataset_match.py    Dataset discovery scoring
    graph.py            BFS graph traversal
  store/
    main.py             SqliteStore — all persistence logic
    search.py           BM25 keyword scorer + cosine similarity
    vec.py              sqlite-vec ANN helpers
docs/
  USAGE.md              End-to-end API usage examples with curl
examples/
  quickstart.py         Full round-trip demo (register → ingest → search)
tests/                  pytest suite (109 tests, no Ollama required)
pyproject.toml          Package metadata + optional extras
.cursor/
  environment.json      Cloud agent install script + background terminals
  mcp.json              stdio MCP config for Cursor IDE
AGENTS.md               ← this file
README.md               User-facing feature summary and quick start
STRATEGY.md             Design rationale (historical; all features now implemented)
ARCHITECTURE_PLAN.md    Architecture decisions and future expansion notes
```

---

## Environment setup

### Install (all platforms)

```bash
pip install -e '.[dev]'
```

`[dev]` pulls in `pytest`, `pytest-asyncio`, `httpx`, and `sqlite-vec`.

If `sqlite-vec` fails on an unusual platform, fall back to:

```bash
pip install -e .
pip install pytest pytest-asyncio httpx
```

Tests still pass; vector search falls back to a Python cosine scan.

### Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CORTEXDB_EMBED_PROVIDER` | `ollama` | `ollama` \| `api` \| `none` |
| `CORTEXDB_EMBED_MODEL` | `nomic-embed-text` | Model name passed to the provider |
| `CORTEXDB_EMBED_URL` | provider-specific | Base URL for the embedding API |
| `CORTEXDB_EMBED_API_KEY` | _(unset)_ | API key for `api` provider |
| `CORTEXDB_OLLAMA_AUTOSTART` | `true` | Auto-start `ollama serve` if not reachable |
| `CORTEXDB_DB_PATH` | `cortexdb.sqlite` | Absolute path to the SQLite file |

Set `CORTEXDB_EMBED_PROVIDER=none` whenever embedding is not needed (tests, registry work, MCP reads). This avoids any Ollama dependency.

---

## Running tests

```bash
export CORTEXDB_EMBED_PROVIDER=none
pytest
```

- 109 tests, all passing.
- Tests spin up a temporary SQLite file; no running server or Ollama required.
- `asyncio_mode = strict` — all async tests must use `@pytest.mark.asyncio`.

Run a single test file:

```bash
pytest tests/test_api.py -v
```

---

## Running the server locally

```bash
CORTEXDB_EMBED_PROVIDER=none uvicorn app.main:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- MCP endpoint: `http://127.0.0.1:8000/mcp` (JSON-RPC POST)
- Health: `http://127.0.0.1:8000/health`

---

`.cursor/environment.json` can define a terminal **CortexDB API** that runs `cortexdb-api --reload` on port 5000 by default with `CORTEXDB_EMBED_PROVIDER=none` for manual checks or HTTP clients inside the VM (`POST /mcp`, `/docs`, etc.). Override with `CORTEXDB_API_PORT`.

## MCP configuration

### stdio (recommended for Cursor Cloud Agents and Claude Desktop)

The `.cursor/mcp.json` in this repo configures stdio MCP automatically when the repo is open in Cursor. For external clients:

```json
{
  "mcpServers": {
    "cortexdb": {
      "command": "python",
      "args": ["-m", "app.mcp.stdio"],
      "env": {
        "CORTEXDB_EMBED_PROVIDER": "none",
        "CORTEXDB_DB_PATH": "/absolute/path/to/cortexdb.sqlite"
      }
    }
  }
}
```

Or use the installed console script:

```bash
CORTEXDB_EMBED_PROVIDER=none CORTEXDB_DB_PATH=/path/to/cortexdb.sqlite cortexdb-mcp
```

### HTTP MCP

Cloud agents cannot reach `localhost` from outside the VM through HTTP MCP. Use stdio MCP inside the VM, or point HTTP MCP at a publicly reachable deployed instance (`POST /mcp`).

### MCP resource URI scheme

| URI | Content |
|---|---|
| `cortexdb://context/index` | Compact orientation index (~50 tokens/item) |
| `cortexdb://graph` | Full relationship map |
| `cortexdb://datasets/{key}` | Full dataset context (schema, examples, relationships) |
| `cortexdb://tools/{key}` | Full tool metadata |

---

## Coding conventions

- Python ≥ 3.10; use `from __future__ import annotations` in all modules.
- All persistence goes through `SqliteStore` in `app/store/main.py`. Do **not** open SQLite connections elsewhere.
- No generative LLM calls anywhere in `app/`. Callers own reasoning; CortexDB owns storage and retrieval.
- Pydantic v2 models for all API schemas (`app/schemas/`).
- FastAPI `Depends()` for store and embedding service injection — never import `_store` directly in route handlers.
- Additive SQLite migrations go in `_MIGRATIONS` list in `app/store/main.py`. They are idempotent (catch `OperationalError` on duplicate columns).
- BM25 and cosine scoring are pure Python in `app/store/search.py` — keep them dependency-free.

---

## Adding a new API endpoint

1. Add Pydantic schema to `app/schemas/` if a new type is needed.
2. Add store methods to `app/store/main.py` (and a migration if schema changes).
3. Write the route handler in the appropriate `app/api/` module (or create a new one).
4. Register the router in `app/api/router.py`.
5. Add tests in `tests/`.
6. Update `docs/USAGE.md` with curl examples.

---

## Adding a new embedding provider

1. Implement the provider class in `app/embed/providers.py`.
2. Register the new provider literal in `EmbedConfig` (`app/embed/config.py`) and wire it in `EmbeddingService.startup()` (`app/embed/service.py`).
3. Document the new `CORTEXDB_EMBED_PROVIDER` value in this file and in `README.md`.

---

## Secrets

Add secrets in [Cloud Agents → Secrets](https://cursor.com/dashboard/cloud-agents). They are injected as environment variables.

| Secret | When needed |
|---|---|
| `CORTEXDB_EMBED_API_KEY` | `CORTEXDB_EMBED_PROVIDER=api` with a key-gated provider |
| `CURSOR_API_KEY` | Programmatic Cursor Cloud SDK runs |

Never commit secrets. Never hard-code them in tests.

---

## Cursor Cloud Agent — quick checklist

Before starting a task:

1. `pip install -e '.[dev]'` (or verify it is already done by the environment install script).
2. `export CORTEXDB_EMBED_PROVIDER=none`
3. `pytest` — all 109 tests must pass before and after your changes.
4. Read the relevant `app/` source files before editing.
5. Run `pytest` again after changes.
6. Commit with a descriptive message; push; open or update the PR.

---

## Known planned work (future, not yet implemented)

- Postgres + pgvector backend (currently SQLite only).
- Tenant / namespace isolation (schema fields exist; enforcement not yet built).
- Scoring profiles stored in DB.
- Re-embedding job scheduler (manual `/re-embed` endpoint exists; no automatic scheduling).

See `ARCHITECTURE_PLAN.md` for full roadmap context.
