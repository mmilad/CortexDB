"""HTTP routes for CortexDB's built-in text processor."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.ingest.analyzer import analyze_ingest
from app.processors.safe import process_text_safe
from app.schemas.ingest_analysis import IngestAnalysisRequest, IngestAnalysisResult
from app.schemas.processor import ProcessorRequest, ProcessorResponse

router = APIRouter(tags=["processor"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cortexdb-processor"}


@router.post("/process/text", response_model=ProcessorResponse)
def process_text(body: ProcessorRequest) -> ProcessorResponse:
    if body.strategy in ("semantic", "extractive"):
        raise HTTPException(
            status_code=501,
            detail=f"processor strategy '{body.strategy}' is not implemented yet",
        )
    try:
        return process_text_safe(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analyze/ingest", response_model=IngestAnalysisResult)
def analyze_ingest_text(body: IngestAnalysisRequest) -> IngestAnalysisResult:
    try:
        return analyze_ingest(
            body.text,
            session_id=body.session_id,
            config=body.config,
            existing_datasets=body.existing_datasets,
            candidate_state=body.candidate_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
