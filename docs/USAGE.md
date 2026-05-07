# CortexDB Usage

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cortexdb-api --reload
```

Storage defaults to `cortexdb.sqlite` in the working directory.
Override with `CORTEXDB_DB_PATH=/path/to/file.sqlite`.
The API port defaults to `5000`; override with `CORTEXDB_API_PORT=5001`.

## Open docs

- Swagger UI: `http://127.0.0.1:5000/docs`
- OpenAPI JSON: `http://127.0.0.1:5000/openapi.json`

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
1. resources/list              → same as context/index but MCP format
2. resources/read cortexdb://datasets/{key}   → full dataset context
3. resources/read cortexdb://graph            → relationship map
4. resources/read cortexdb://context/index    → minimal orientation
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

### Discover or get a blueprint for a new dataset

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

---

## Ingest Pipeline

CortexDB includes a reusable deterministic ingest pipeline for preparing text,
Markdown, and plain text files for the existing dataset ingest path. Callers
still provide raw content only; CortexDB owns embedding and storage. Actual
ingest requires an enabled embedding provider.

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

### BFS from a starting node (depth 1-5)

```bash
curl "http://127.0.0.1:5000/graph/explore?start=dataset:tech_knowledge&depth=2"
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
