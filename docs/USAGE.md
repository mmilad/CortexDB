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
