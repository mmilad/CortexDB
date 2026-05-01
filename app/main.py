from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="CortexDB API",
    version="0.1.0",
    description=(
        "CortexDB service API (no internal LLM logic). "
        "Provides deterministic memory, registry, and retrieval metadata interfaces."
    ),
)


class DatasetRecord(BaseModel):
    dataset_key: str = Field(..., description="Stable dataset identifier")
    display_name: str
    schema_version: str
    semantic_description: str
    usage_guidance: str
    relationship_hints: list[str] = Field(default_factory=list)
    filterable_fields: list[str] = Field(default_factory=list)
    status: str = "active"


class ToolRecord(BaseModel):
    tool_key: str = Field(..., description="Stable tool identifier")
    name: str
    description: str
    capability_tags: list[str] = Field(default_factory=list)
    relationship_hints: list[str] = Field(default_factory=list)
    embedding_model_version: str | None = None
    status: str = "active"


DATASETS: dict[str, dict[str, Any]] = {}
TOOLS: dict[str, dict[str, Any]] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "service": "cortexdb",
        "llm_inside": False,
        "resources": {
            "datasets": list(DATASETS.keys()),
            "tools": list(TOOLS.keys()),
        },
        "docs": {"swagger_ui": "/docs", "openapi_json": "/openapi.json"},
    }


@app.post("/datasets", response_model=DatasetRecord)
def upsert_dataset(record: DatasetRecord) -> DatasetRecord:
    DATASETS[record.dataset_key] = record.model_dump()
    return record


@app.get("/datasets", response_model=list[DatasetRecord])
def list_datasets() -> list[DatasetRecord]:
    return [DatasetRecord(**d) for d in DATASETS.values()]


@app.get("/datasets/{dataset_key}", response_model=DatasetRecord)
def get_dataset(dataset_key: str) -> DatasetRecord:
    data = DATASETS.get(dataset_key)
    if not data:
        raise HTTPException(status_code=404, detail="dataset not found")
    return DatasetRecord(**data)


@app.post("/tools", response_model=ToolRecord)
def upsert_tool(record: ToolRecord) -> ToolRecord:
    TOOLS[record.tool_key] = record.model_dump()
    return record


@app.get("/tools", response_model=list[ToolRecord])
def list_tools() -> list[ToolRecord]:
    return [ToolRecord(**t) for t in TOOLS.values()]


@app.get("/tools/{tool_key}", response_model=ToolRecord)
def get_tool(tool_key: str) -> ToolRecord:
    tool = TOOLS.get(tool_key)
    if not tool:
        raise HTTPException(status_code=404, detail="tool not found")
    return ToolRecord(**tool)
