from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.dataset import DatasetRecord, RetrievalCapability


class DatasetDiscoverRequest(BaseModel):
    intent: str = Field(
        ...,
        description="Natural language description of storage or query needs.",
    )
    required_capabilities: list[RetrievalCapability] = Field(
        default_factory=list,
        description="Dataset must declare all of these in retrieval_capabilities.",
    )
    content_kind: str | None = Field(
        default=None,
        description="Optional preferred content_kind (e.g. documents).",
    )
    tag_filters: list[str] = Field(
        default_factory=list,
        description="Boost score when dataset capability_tags overlap (any match).",
    )


class DatasetCandidate(BaseModel):
    dataset: DatasetRecord
    score: float
    reasons: list[str]


class DatasetBlueprint(BaseModel):
    """Suggested payload for POST /datasets when recommended_action is create_new."""

    suggested_dataset_key: str = Field(
        ...,
        description="Proposed stable key; caller may change before create.",
    )
    record: DatasetRecord


class DatasetDiscoverResponse(BaseModel):
    candidates: list[DatasetCandidate]
    recommended_action: Literal["use_existing", "create_new"]
    suggested_blueprint: DatasetBlueprint | None = Field(
        default=None,
        description="Populated when recommended_action is create_new.",
    )
