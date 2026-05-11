"""Validation for processor output before CortexDB persists anything."""

from __future__ import annotations

from app.schemas.processor import ProcessorResponse, ProcessorSpan


class ProcessorValidationError(ValueError):
    """Raised when a processor returns unsafe or inconsistent spans."""


def _validate_span_text(source: str, span: ProcessorSpan, *, max_chars: int) -> None:
    if span.char_end <= span.char_start:
        raise ProcessorValidationError("processor span has empty or negative range")
    if span.char_end > len(source):
        raise ProcessorValidationError("processor span is outside source text")
    if not span.text.strip():
        raise ProcessorValidationError("processor span text is empty")
    if len(span.text) > max_chars:
        raise ProcessorValidationError("processor span exceeds max_chars")
    if source[span.char_start:span.char_end] != span.text:
        raise ProcessorValidationError("processor span text does not match source offsets")


def validate_processor_response(
    source: str,
    response: ProcessorResponse,
    *,
    max_chars: int,
) -> ProcessorResponse:
    if not response.chunks:
        raise ProcessorValidationError("processor returned no chunks")

    for chunk in response.chunks:
        _validate_span_text(source, chunk, max_chars=max_chars)

    for primitive in response.primitives:
        if primitive.char_end <= primitive.char_start:
            raise ProcessorValidationError("processor primitive has empty or negative range")
        if primitive.char_end > len(source):
            raise ProcessorValidationError("processor primitive is outside source text")
        if not primitive.text.strip():
            raise ProcessorValidationError("processor primitive text is empty")
        if source[primitive.char_start:primitive.char_end] != primitive.text:
            raise ProcessorValidationError("processor primitive text does not match source offsets")

    return response
