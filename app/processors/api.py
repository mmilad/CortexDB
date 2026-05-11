"""FastAPI app for the optional CortexDB processor sidecar."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.ingest.analyzer import analyze_ingest
from app.schemas.ingest_analysis import IngestAnalysisRequest, IngestAnalysisResult
from app.schemas.processor import ProcessorRequest, ProcessorResponse
from app.processors.safe import process_text_safe

app = FastAPI(
    title="CortexDB Processor",
    description="Optional long-lived text processor sidecar for CortexDB.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cortexdb-processor"}


@app.post("/process/text", response_model=ProcessorResponse)
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


@app.post("/analyze/ingest", response_model=IngestAnalysisResult)
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
