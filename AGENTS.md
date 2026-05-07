# CortexDB — agent notes

## Cursor Cloud Agents

### Environment

Cloud agents use `.cursor/environment.json` in this repo. The install step runs:

`pip install -e '.[dev]'` (includes test dependencies and optional `sqlite-vec` for ANN search).

If `sqlite-vec` fails to install on an unusual platform, use `pip install -e .` plus `pip install pytest pytest-asyncio httpx` and run tests without the vec extra.

### Quick verification

From the repository root (with dependencies installed):

```bash
export CORTEXDB_EMBED_PROVIDER=none
pytest
```

Tests use a temporary SQLite file; they do not require Ollama or a running server.

### MCP for Cloud Agents

Cloud agents cannot reach `localhost` on your laptop through **HTTP** MCP: the Cursor backend proxies HTTP MCP to a URL you provide, so that URL must be **publicly reachable** (for example a deployed CortexDB instance ending in `/mcp`).

For work against **this checkout** inside the cloud VM, prefer **stdio** MCP (runs in the agent environment after `install`):

1. Open [cursor.com/agents](https://cursor.com/agents) → MCP → add a custom server.
2. Use **stdio** transport (HTTP is fine too if you operate a public `POST /mcp` endpoint).
3. Suggested command: `python3` (or `python`) with arguments `-m`, `app.mcp.stdio`.
4. Working directory: repository root (same folder as `pyproject.toml`).
5. Environment variables:
   - `CORTEXDB_EMBED_PROVIDER=none` — avoids Ollama startup when you only need registry/MCP reads.
   - `CORTEXDB_DB_PATH` — absolute path to the SQLite file the agent should use (for example under the cloned workspace). Omit to use the default file in the process working directory.

Registry-oriented MCP calls do not require embedding. Enable embedding secrets only if the agent must call ingest/search against a live provider (see README for `CORTEXDB_EMBED_*` variables).

### Optional: local API during a run

`.cursor/environment.json` can define a terminal **CortexDB API** that runs `cortexdb-api --reload` on port 5000 by default with `CORTEXDB_EMBED_PROVIDER=none` for manual checks or HTTP clients inside the VM (`POST /mcp`, `/docs`, etc.). Override with `CORTEXDB_API_PORT`.

### Secrets

Add provider keys (for example `CORTEXDB_EMBED_API_KEY`) in [Cloud Agents → Secrets](https://cursor.com/dashboard/cloud-agents) when tests or tasks need real embedding or external APIs. Do not commit secrets.

### API keys for Cursor Cloud / SDK

For programmatic runs, create a key under [Cloud Agents](https://cursor.com/dashboard/cloud-agents) and use `CURSOR_API_KEY` in your environment. For TypeScript automation against cloud runs, pass `cloud: { repos: [...] }` explicitly so the runtime is not accidentally local.
