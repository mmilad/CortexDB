from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.tool import ToolRecord
from app.state import RegistryState, get_registry

router = APIRouter(tags=["tools"])


@router.post("/tools", response_model=ToolRecord)
def upsert_tool(
    record: ToolRecord,
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> ToolRecord:
    reg.tools[record.tool_key] = record.model_dump()
    return record


@router.get("/tools", response_model=list[ToolRecord])
def list_tools(
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> list[ToolRecord]:
    return [ToolRecord(**t) for t in reg.tools.values()]


@router.get("/tools/{tool_key}", response_model=ToolRecord)
def get_tool(
    tool_key: str,
    reg: Annotated[RegistryState, Depends(get_registry)],
) -> ToolRecord:
    tool = reg.tools.get(tool_key)
    if not tool:
        raise HTTPException(status_code=404, detail="tool not found")
    return ToolRecord(**tool)
