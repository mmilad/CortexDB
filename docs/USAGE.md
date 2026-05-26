# CortexDB Usage

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[vec]"     # includes sqlite-vec for ANN search
cortexdb-api --reload
```

Storage defaults to `cortexdb.sqlite` in the working directory.
Override with `CORTEXDB_DB_PATH=/path/to/file.sqlite`.
The API port defaults to `5000`; override with `CORTEXDB_API_PORT=5001`.

Disable embedding (faster startup, registry + MCP reads still work):

```bash
CORTEXDB_EMBED_PROVIDER=none cortexdb-api --reload
```

## Open docs

- Swagger UI: `http://127.0.0.1:5000/docs`
- OpenAPI JSON: `http://127.0.0.1:5000/openapi.json`
- Health check: `http://127.0.0.1:5000/health`

The safe deterministic processor is integrated into the main API by default:

- `POST /processor/process/text`
- `POST /processor/analyze/ingest`
- `GET /processor/health`

Set `CORTEXDB_PROCESSOR_CLIENT_PROVIDER=sidecar` and run `cortexdb-processor`
only when you intentionally want a separate processor process.

---

## LLM Agent Workflow (Recommended)

An LLM agent should follow this pattern for low token usage:

```
1. GET /context/index          → orient: what datasets and tools exist? (~50 tokens/item)
2. GET /context/dataset/{key}  → deep context for relevant datasets (~300 tokens each)
3. GET /context/graph          → understand relationships (~60 tokens/edge)
4. GET /graph/explore?start=.. → BFS subgraph from a starting node
```

For MCP-native agents:

```
1. resources/list                               → same as context/index but MCP format
2. resources/read cortexdb://context/index      → minimal orientation
3. resources/read cortexdb://datasets/{key}     → full dataset context
4. resources/read cortexdb://graph              → relationship map
```

---

## Datasets

### Create / update a dataset (rich LLM metadata)

```bash
curl -X POST http://127.0.0.1:5000/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "tech_knowledge",
    "display_name": "Tech Knowledge Base",
    "schema_version": "v1",
    "semantic_description": "Technical KB articles for engineering Q&A",
    "usage_guidance": "Query when resolving engineering issues or finding how-to guides",
    "llm_summary": "Engineering KB with runbooks and troubleshooting guides. Query by component or severity.",
    "retrieval_capabilities": ["vector", "keyword"],
    "content_kind": "documents",
    "capability_tags": ["rag", "engineering"],
    "entity_types": ["KBArticle", "Runbook"],
    "access_patterns": ["by_component", "semantic_search"],
    "filterable_fields": ["component", "severity"],
    "field_descriptions": [
      {"field": "component", "description": "Engineering component", "example_values": ["api", "database"]},
      {"field": "severity", "description": "Relevance severity", "example_values": ["critical", "high"]}
    ],
    "query_examples": [
      {"label": "by_component", "description": "Find articles for a component", "example_request": {"filters": {"component": "database"}}},
      {"label": "semantic_search", "description": "Vector search", "example_request": {"query": "connection pool exhaustion", "retrieval": "vector"}}
    ],
    "status": "active"
  }'
```

### List all datasets

```bash
curl http://127.0.0.1:5000/datasets
```

### Get a single dataset

```bash
curl http://127.0.0.1:8000/datasets/tech_knowledge
```

### Delete a dataset

Deletes the dataset record and cascades to its memory items and relationships.

```bash
curl -X DELETE http://127.0.0.1:8000/datasets/tech_knowledge
```

### Discover or match a dataset by intent

```bash
curl -X POST http://127.0.0.1:5000/datasets/discover \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "store engineering runbooks for semantic search",
    "required_capabilities": ["vector"],
    "content_kind": "documents",
    "tag_filters": ["rag"]
  }'
```

### Validate a dataset (stamp last_validated_at)

Call this after verifying the dataset metadata is consistent with the backing data. The `last_validated_at` timestamp can be monitored to detect metadata drift.

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/validate
```

---

## Memory items (ingest, search, list, delete)

Memory items are raw-text records stored inside a dataset. CortexDB embeds them automatically.

### Ingest raw text items

Embedding must be enabled (`CORTEXDB_EMBED_PROVIDER=ollama` or `=api`).

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"raw_text": "Connection pool exhaustion fix: increase pool size in config.", "metadata": {"component": "database", "severity": "critical"}},
      {"raw_text": "Auth service returns 401 when JWT secret is rotated.", "metadata": {"component": "auth", "severity": "high"}}
    ]
  }'
```

### Vector search

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection issues", "top_k": 5}'
```

### Keyword-only search (no embedding required)

Set `vector_weight=0.0` and provide `keyword_query`. Works even when `CORTEXDB_EMBED_PROVIDER=none`.

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "", "keyword_query": "connection pool", "vector_weight": 0.0, "top_k": 5}'
```

### Hybrid search (vector + keyword blend)

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection issues", "keyword_query": "pool exhaustion", "vector_weight": 0.7, "top_k": 5}'
```

### Search with metadata filters

```bash
curl -X POST http://127.0.0.1:8000/datasets/tech_knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "auth failures", "metadata_filters": {"component": "auth"}, "top_k": 5}'
```

### List memory items (paginated)

```bash
curl "http://127.0.0.1:8000/datasets/tech_knowledge/items?limit=20&offset=0"
```

Include soft-deleted items:

```bash
curl "http://127.0.0.1:8000/datasets/tech_knowledge/items?include_deleted=true"
```

### Get a single memory item

```bash
curl http://127.0.0.1:8000/datasets/tech_knowledge/items/{item_id}
```

### Soft-delete a memory item (recoverable)

The item is excluded from queries by default but can be retrieved with `include_deleted=true`.

```bash
curl -X DELETE http://127.0.0.1:8000/datasets/tech_knowledge/items/{item_id}
```

### Hard-delete a memory item (irreversible)

Permanently removes the row and its vec0 ANN entry. Also deletes any relationships referencing this item.

```bash
curl -X DELETE http://127.0.0.1:8000/datasets/tech_knowledge/items/{item_id}/hard
```

### Re-embed all items in a dataset

Use after changing the embedding model to re-vectorize all stored items. Rebuilds the vec0 ANN index automatically.

```bash
curl -X POST "http://127.0.0.1:8000/datasets/tech_knowledge/re-embed?batch_size=50"
```

---

## Ingest Pipeline

CortexDB includes a reusable deterministic ingest pipeline for preparing text,
Markdown, and plain text files for the existing dataset ingest path. Callers
still provide raw content only; CortexDB owns embedding and storage. Actual
ingest requires an enabled embedding provider.

### Session-aware ingest front door

Use `POST /ingest` when CortexDB should act as middleware for a chat,
assistant, or A2A workflow. This endpoint always stores both chat history and
an auditable raw text record. Derived work such as summaries, facts, decisions,
goals, and knowledge extraction is reported separately and never blocks the
durable session/raw write.

```bash
curl -X POST http://127.0.0.1:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "main",
    "role": "user",
    "text": "Remember that the API worker must restart after credential rotation.",
    "source": "user_prompt",
    "dataset_policy": "create_if_needed",
    "metadata": {"app": "assistant-ui"}
  }'
```

If `session_id` is omitted, CortexDB uses the default `main` session. Session
scope defaults to the current namespace. Set `scope_mode` to `global` or
`explicit` when the consuming app wants a different retrieval boundary.

Configure optional LLM extraction with an OpenAI-compatible endpoint:

```bash
CORTEXDB_LLM_PROVIDER=api \
CORTEXDB_LLM_URL=https://api.openai.com \
CORTEXDB_LLM_MODEL=gpt-4.1-mini \
CORTEXDB_LLM_API_KEY=sk-... \
cortexdb-api --reload
```

Without an LLM provider, `/ingest` still writes `sessions`,
`session_messages`, and `raw_texts`; the derived jobs return `skipped`.

Derived extraction uses structured JSON output. CortexDB sends a JSON Schema in
`response_format` and validates the model response before writing memory items.
The generic envelope is:

```json
{
  "schema_version": "cortexdb.derived_memory.v1",
  "memories": [
    {
      "dataset_key": "derived_preferences",
      "kind": "preference",
      "text": "User prefers local Ollama models for development.",
      "score": 0.9,
      "metadata": {"scope": "dev"},
      "dataset": {
        "display_name": "Derived Preferences",
        "semantic_description": "User and workflow preferences extracted from ingest.",
        "usage_guidance": "Use when adapting assistant behavior to user preferences.",
        "entity_types": ["Preference"],
        "capability_tags": ["derived", "preferences"]
      }
    }
  ]
}
```

Fetch chat history and prompt-ready context:

```bash
curl http://127.0.0.1:5000/sessions/main/history

curl -X POST http://127.0.0.1:5000/context \
  -H "Content-Type: application/json" \
  -d '{"session_id": "main", "prompt": "credential rotation restart", "top_k": 5}'
```

Rename a chat title, rename its session id, or delete the chat:

```bash
curl -X PATCH http://127.0.0.1:5000/sessions/main \
  -H "Content-Type: application/json" \
  -d '{"title": "Project notes"}'

curl -X PATCH http://127.0.0.1:5000/sessions/main \
  -H "Content-Type: application/json" \
  -d '{"id": "project_notes"}'

curl -X DELETE http://127.0.0.1:5000/sessions/project_notes

curl -X DELETE "http://127.0.0.1:5000/sessions/project_notes?delete_related_chunks=true"
```

By default, deleting a session removes the chat shell and message history but
keeps raw texts and dataset-backed memory. Set `delete_related_chunks=true` for
a deeper cleanup that also removes related raw texts and memory chunks or
observations linked by `session_id`, `raw_text_id`, or `session_message_id`.

### Build chunks from a Markdown file

```python
from pathlib import Path

from app.ingest import build_ingest_items

items = build_ingest_items(
    Path("docs/runbook.md"),
    max_chars=2000,
    overlap_chars=200,
    metadata={"component": "api"},
)

for item in items:
    print(item.id, item.metadata["chunk_index"], item.raw_text[:80])
```

Each generated item is compatible with `POST /datasets/{key}/ingest` and
includes metadata such as `source_type`, `source_path`, `filename`,
`chunk_index`, `chunk_count`, content/source SHA-256 hashes, and `ingestion_id`.

### Ingest a directory from Python

```python
from pathlib import Path

from app.embed.service import get_embedding_service
from app.ingest import ingest_directory_to_dataset
from app.store import get_store

result = await ingest_directory_to_dataset(
    "tech_knowledge",
    Path("docs"),
    get_store(),
    get_embedding_service(),
    max_chars=2000,
    overlap_chars=200,
    batch_size=100,
)
print(result.ingested, result.ids)
```

Directory traversal is recursive and deterministic. Only `.txt` and `.md`
files are included; unsupported files are ignored.

### Chunk and ingest text over HTTP

```bash
curl -X POST http://127.0.0.1:5000/datasets/tech_knowledge/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# Runbook\n\nRestart the API worker after rotating credentials.",
    "metadata": {"component": "api", "kind": "runbook"},
    "max_chars": 2000,
    "overlap_chars": 200,
    "batch_size": 100
  }'
```

Do not send vectors to CortexDB. The pipeline only prepares raw text chunks and
metadata, then delegates to the existing ingest service for embedding.

---

## Tools

### Create / update a tool

```bash
curl -X POST http://127.0.0.1:5000/tools \
  -H "Content-Type: application/json" \
  -d '{
    "tool_key": "log_search",
    "name": "Log Search",
    "description": "Search logs by metadata and semantic hints",
    "llm_summary": "Search production logs by component, level, or semantic query.",
    "capability_tags": ["observability", "search"],
    "relationship_hints": ["tech_knowledge"],
    "safety_scope": "read-only",
    "query_examples": [
      {"label": "by_level", "description": "Find error logs", "example_input": {"component": "auth", "level": "ERROR"}},
      {"label": "semantic", "description": "Find by symptom", "example_input": {"query": "connection refused"}}
    ],
    "embedding_model_version": "text-embed-v1",
    "status": "active"
  }'
```

### List all tools

```bash
curl http://127.0.0.1:8000/tools
```

### Get a single tool

```bash
curl http://127.0.0.1:8000/tools/log_search
```

### Delete a tool

```bash
curl -X DELETE http://127.0.0.1:8000/tools/log_search
```

---

## Relationships

### Declare a typed edge between nodes

```bash
curl -X POST http://127.0.0.1:5000/relationships \
  -H "Content-Type: application/json" \
  -d '{
    "id": "rel-001",
    "source_type": "dataset",
    "source_key": "known_issues",
    "target_type": "dataset",
    "target_key": "tech_knowledge",
    "edge_type": "related",
    "description": "Issues reference KB articles for workarounds",
    "join_fields": []
  }'
```

Edge types: `joins_on`, `feeds_into`, `shared_entity`, `produces`, `consumes`, `related`.

### List edges touching a node

```bash
curl "http://127.0.0.1:5000/relationships?node_key=tech_knowledge"
```

### Get a single relationship

```bash
curl http://127.0.0.1:8000/relationships/rel-001
```

### Delete a relationship

```bash
curl -X DELETE http://127.0.0.1:8000/relationships/rel-001
```

---

## LLM Context Endpoints

### Minimal orientation index (call this first)

```bash
curl http://127.0.0.1:5000/context/index
```

### Full context for one dataset (query guidance + examples)

```bash
curl http://127.0.0.1:5000/context/dataset/tech_knowledge
```

### Full context for one tool

```bash
curl http://127.0.0.1:5000/context/tool/log_search
```

### Compact relationship map

```bash
curl http://127.0.0.1:5000/context/graph
```

---

## Graph Traversal

### BFS from a starting node (depth 1–5)

```bash
curl "http://127.0.0.1:5000/graph/explore?start=dataset:tech_knowledge&depth=2"
```

Response shape:

```json
{
  "nodes": [{"key": "tech_knowledge", "type": "dataset"}, ...],
  "edges": [{"source": "tech_knowledge", "target": "known_issues", "edge_type": "related", ...}]
}
```

---

## MCP Endpoint

CortexDB exposes a dynamic MCP JSON-RPC endpoint at `/mcp`.
All datasets and tools registered in the registry are automatically
discoverable — no code changes or restarts needed when new items are added.

### Initialize

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"my-agent"}}}'
```

### List all resources (datasets + tools + static resources)

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}'
```

### Read context index (minimal orientation)

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"cortexdb://context/index"}}'
```

### Read full dataset context

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"cortexdb://datasets/tech_knowledge"}}'
```

### Read relationship graph

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{"uri":"cortexdb://graph"}}'
```

### List MCP tools

```bash
curl -X POST http://127.0.0.1:5000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":6,"method":"tools/list","params":{}}'
```

### Call an MCP tool (passthrough — execution is external)

CortexDB does not execute tools internally. `tools/call` returns the tool's registry metadata so the caller can execute it.

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"log_search","arguments":{}}}'
```
