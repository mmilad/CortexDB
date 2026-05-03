from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.tool import ToolRecord
from app.store import SqliteStore, get_store

router = APIRouter(tags=["tools"])


@router.post("/tools", response_model=ToolRecord)
def upsert_tool(
    record: ToolRecord,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> ToolRecord:
    store.upsert_tool(record.tool_key, record.model_dump())
    return record


@router.get("/tools", response_model=list[ToolRecord])
def list_tools(
    store: Annotated[SqliteStore, Depends(get_store)],
) -> list[ToolRecord]:
    return [ToolRecord(**t) for t in store.list_tools().values()]


@router.get("/tools/{tool_key}", response_model=ToolRecord)
def get_tool(
    tool_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> ToolRecord:
    data = store.get_tool(tool_key)
    if not data:
        raise HTTPException(status_code=404, detail="tool not found")
    return ToolRecord(**data)


@router.delete(
    "/tools/{tool_key}",
    summary="Delete a tool",
    description="Removes the tool record from the registry.",
)
def delete_tool(
    tool_key: str,
    store: Annotated[SqliteStore, Depends(get_store)],
) -> dict:
    deleted = store.delete_tool(tool_key)
    if not deleted:
        raise HTTPException(status_code=404, detail="tool not found")
    return {"deleted": tool_key}
