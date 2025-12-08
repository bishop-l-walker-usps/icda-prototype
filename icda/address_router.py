"""FastAPI router for address verification endpoints.

This module provides the REST API endpoints for the address verification
pipeline, including single address verification, batch processing,
and index management.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from icda.address_models import (
    BatchItem,
    VerificationStatus,
)
from icda.address_pipeline import AddressPipeline, BatchProcessor


logger = logging.getLogger(__name__)

# Router instance - will be configured with pipeline in main.py
router = APIRouter(prefix="/api/address", tags=["Address Verification"])

# Global reference to pipeline (set during app startup)
_pipeline: AddressPipeline | None = None
_batch_processor: BatchProcessor | None = None


def configure_router(pipeline: AddressPipeline) -> None:
    """Configure the router with pipeline dependencies.

    Args:
        pipeline: Initialized address verification pipeline.
    """
    global _pipeline, _batch_processor
    _pipeline = pipeline
    _batch_processor = BatchProcessor(pipeline)
    logger.info("Address router configured")


# ============================================================================
# Request/Response Models
# ============================================================================


class AddressVerifyRequest(BaseModel):
    """Request model for single address verification."""

    address: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw address string to verify",
        examples=["101 turkey ok 22222", "123 Main St, New York, NY 10001"],
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Optional context hints (zip, state, etc.)",
    )


class AddressVerifyResponse(BaseModel):
    """Response model for single address verification."""

    success: bool
    status: str
    original: dict[str, Any]
    verified: dict[str, Any] | None
    confidence: float
    match_type: str | None
    alternatives: list[dict[str, Any]]
    processing_time_ms: int
    metadata: dict[str, Any]


class BatchVerifyRequest(BaseModel):
    """Request model for batch address verification."""

    addresses: list[str] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of addresses to verify",
    )
    concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent verifications",
    )


class BatchRecordRequest(BaseModel):
    """Request model for batch verification with record format."""

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of records with address data",
    )
    address_field: str = Field(
        default="address",
        description="Field name containing address",
    )
    id_field: str = Field(
        default="id",
        description="Field name for unique ID",
    )
    concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent verifications",
    )


class BatchVerifyResponse(BaseModel):
    """Response model for batch verification."""

    success: bool
    results: list[dict[str, Any]]
    summary: dict[str, Any]


class StreetSuggestionRequest(BaseModel):
    """Request model for street name suggestions."""

    partial: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Partial street name",
    )
    zip_code: str = Field(
        ...,
        min_length=5,
        max_length=5,
        description="5-digit ZIP code",
    )
    street_number: str | None = Field(
        default=None,
        description="Optional street number for context",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum suggestions to return",
    )


class StreetSuggestionResponse(BaseModel):
    """Response model for street suggestions."""

    suggestions: list[dict[str, Any]]
    zip_code: str
    partial: str


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""

    total_addresses: int
    unique_zips: int
    unique_states: int
    unique_cities: int
    unique_streets: int
    indexed: bool


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/verify", response_model=AddressVerifyResponse)
async def verify_address(request: AddressVerifyRequest) -> AddressVerifyResponse:
    """Verify a single address through the pipeline.

    This endpoint processes the address through all verification stages:
    1. Normalize and parse the address
    2. Classify quality and identify issues
    3. Look for exact matches in known addresses
    4. Find fuzzy matches if no exact match
    5. Use Nova AI to complete/correct if needed

    Args:
        request: Address verification request.

    Returns:
        Verification result with status, confidence, and alternatives.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = await _pipeline.verify(request.address)

    return AddressVerifyResponse(
        success=result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.CORRECTED,
            VerificationStatus.COMPLETED,
        ),
        status=result.status.value,
        original=result.original.to_dict(),
        verified=result.verified.to_dict() if result.verified else None,
        confidence=result.confidence,
        match_type=result.match_type,
        alternatives=[a.to_dict() for a in result.alternatives],
        processing_time_ms=result.metadata.get("total_time_ms", 0),
        metadata=result.metadata,
    )


@router.post("/verify/batch", response_model=BatchVerifyResponse)
async def verify_batch(request: BatchVerifyRequest) -> BatchVerifyResponse:
    """Verify a batch of addresses.

    Processes multiple addresses concurrently with configurable
    parallelism. Returns individual results and summary statistics.

    Args:
        request: Batch verification request.

    Returns:
        Results for each address and overall summary.
    """
    if not _batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not initialized")

    results, summary = await _batch_processor.process_list(
        request.addresses,
        concurrency=request.concurrency,
    )

    return BatchVerifyResponse(
        success=summary.success_rate > 0.5,
        results=[r.to_dict() for r in results],
        summary=summary.to_dict(),
    )


@router.post("/verify/records", response_model=BatchVerifyResponse)
async def verify_records(request: BatchRecordRequest) -> BatchVerifyResponse:
    """Verify addresses from structured records.

    Accepts records with flexible field names for address data.
    Useful for processing data exports or database dumps.

    Args:
        request: Batch record verification request.

    Returns:
        Results for each record and overall summary.
    """
    if not _batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not initialized")

    results, summary = await _batch_processor.process_records(
        request.records,
        address_field=request.address_field,
        id_field=request.id_field,
        concurrency=request.concurrency,
    )

    return BatchVerifyResponse(
        success=summary.success_rate > 0.5,
        results=[r.to_dict() for r in results],
        summary=summary.to_dict(),
    )


@router.post("/suggest/street", response_model=StreetSuggestionResponse)
async def suggest_street(request: StreetSuggestionRequest) -> StreetSuggestionResponse:
    """Get street name suggestions for a partial input.

    Useful for autocomplete functionality or manual address
    correction. Returns street names within the specified ZIP
    that match the partial input.

    Example: partial="turkey", zip="22222" -> ["Turkey Run", "Turkey Trot Ln"]

    Args:
        request: Street suggestion request.

    Returns:
        List of matching street name suggestions.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    suggestions = await _pipeline.completer.suggest_street_completion(
        request.partial,
        request.zip_code,
        request.street_number,
    )

    return StreetSuggestionResponse(
        suggestions=suggestions[:request.limit],
        zip_code=request.zip_code,
        partial=request.partial,
    )


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats() -> IndexStatsResponse:
    """Get address index statistics.

    Returns information about the known address index including
    total addresses indexed and unique counts.

    Returns:
        Index statistics.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = _pipeline.index.stats()

    return IndexStatsResponse(
        total_addresses=stats["total_addresses"],
        unique_zips=stats["unique_zips"],
        unique_states=stats["unique_states"],
        unique_cities=stats["unique_cities"],
        unique_streets=stats["unique_streets"],
        indexed=stats["indexed"],
    )


@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger a rebuild of the address index.

    Rebuilds the index from the current customer database.
    This operation runs in the background.

    Returns:
        Status message.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # For now, just return status - in production you'd queue this
    return {
        "status": "Index rebuild would be triggered here",
        "note": "Current implementation builds index on startup",
    }


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Check address verification service health.

    Returns:
        Health status of the address verification components.
    """
    return {
        "status": "healthy" if _pipeline else "not_initialized",
        "pipeline": _pipeline is not None,
        "batch_processor": _batch_processor is not None,
        "index_ready": _pipeline.index.is_indexed if _pipeline else False,
        "completer_available": _pipeline.completer.available if _pipeline else False,
    }
