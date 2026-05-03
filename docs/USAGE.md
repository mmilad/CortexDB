# CortexDB Usage (FastAPI + Swagger)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

## Open docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Example calls

Create dataset:

```bash
curl -X POST http://127.0.0.1:8000/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "tech_knowledge",
    "display_name": "Tech Knowledge",
    "schema_version": "v1",
    "semantic_description": "Technical KB",
    "usage_guidance": "Use for engineering Q&A",
    "relationship_hints": ["known_issues"],
    "filterable_fields": ["component", "severity"],
    "status": "active"
  }'
```

Create a document-oriented dataset that declares vector retrieval (optional metadata fields):

```bash
curl -X POST http://127.0.0.1:8000/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "documents",
    "display_name": "Documents",
    "schema_version": "v1",
    "semantic_description": "Long-form documents and chunks for RAG",
    "usage_guidance": "Use for semantic similarity over ingested document chunks",
    "content_kind": "documents",
    "retrieval_capabilities": ["vector", "filter_only"],
    "capability_tags": ["rag", "documents"],
    "relationship_hints": [],
    "filterable_fields": ["source_id", "mime_type"],
    "table_refs": [],
    "retrieval_profiles": [{"name": "default_memory", "description": "Vector-heavy preset"}],
    "metadata": {},
    "status": "active"
  }'
```

Discover whether an existing dataset fits, or get a suggested blueprint for `POST /datasets`:

```bash
curl -X POST http://127.0.0.1:8000/datasets/discover \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "store engineering runbooks for semantic search",
    "required_capabilities": ["vector"],
    "content_kind": "documents",
    "tag_filters": ["rag"]
  }'
```

Create tool:

```bash
curl -X POST http://127.0.0.1:8000/tools \
  -H "Content-Type: application/json" \
  -d '{
    "tool_key": "log_search",
    "name": "Log Search",
    "description": "Search logs by metadata and vector hints",
    "capability_tags": ["observability", "search"],
    "relationship_hints": ["tech_knowledge"],
    "embedding_model_version": "text-embed-v1",
    "status": "active"
  }'
```
