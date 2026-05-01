# CortexDB

CortexDB is a service-first memory/retrieval layer for AI/LLM/agentic systems.

## Core principle

- **No LLM logic inside CortexDB.**
- CortexDB stores, indexes, filters, and scores data.
- Consumer applications provide embeddings/intents and perform reasoning externally.

## Python + venv recommendation

Yes — Python with a virtual environment is a good choice for this project (especially for fast API iteration and ecosystem support).

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## FastAPI usage (with Swagger)

Run the API:

```bash
uvicorn app.main:app --reload
```

Open docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Docs

- Architecture plan: [`ARCHITECTURE_PLAN.md`](./ARCHITECTURE_PLAN.md)
- Usage guide: [`docs/USAGE.md`](./docs/USAGE.md)
