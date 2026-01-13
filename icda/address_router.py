"""FastAPI router for address verification endpoints.

This module provides the REST API endpoints for the address verification
pipeline, including single address verification, batch processing,
and index management.

Now includes the 5-agent architecture endpoints for intelligent
address inference with context awareness and multi-state support.
"""

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, ConfigDict

from icda.address_models import (
    BatchItem,
    VerificationStatus,
)
from icda.address_pipeline import AddressPipeline, BatchProcessor
from icda.address_validator_engine import (
    AddressValidatorEngine,
    ValidationMode,
    ValidationResult,
)
from icda.utils.resilience import (
    ValidationAuditEntry,
    validation_audit_log,
)

from icda.agents.orchestrator import AddressAgentOrchestrator


logger = logging.getLogger(__name__)

# Router instance - will be configured with pipeline in main.py
router = APIRouter(
    prefix="/api/address",
    tags=["Address Verification"],
    responses={
        503: {"description": "Service not initialized - pipeline or validator engine unavailable"},
    },
)

# Global reference to pipeline (set during app startup)
_pipeline: AddressPipeline | None = None
_batch_processor: BatchProcessor | None = None
_orchestrator = None  # Will be AddressAgentOrchestrator when implemented
_validator_engine: AddressValidatorEngine | None = None


def configure_router(
    pipeline: AddressPipeline,
    orchestrator=None,
) -> None:
    """Configure the router with pipeline dependencies.

    Args:
        pipeline: Initialized address verification pipeline.
        orchestrator: Optional 5-agent orchestrator for intelligent verification.
    """
    global _pipeline, _batch_processor, _orchestrator, _validator_engine
    _pipeline = pipeline
    _batch_processor = BatchProcessor(pipeline)
    _orchestrator = orchestrator
    _validator_engine = AddressValidatorEngine(address_index=pipeline.index)
    logger.info(f"Address router configured (orchestrator={'enabled' if orchestrator else 'disabled'})")


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
    # Puerto Rico specific fields
    is_puerto_rico: bool = Field(
        default=False,
        description="True if address is Puerto Rico (ZIP 006-009)",
    )
    urbanization: str | None = Field(
        default=None,
        description="Puerto Rico urbanization name (URB) if applicable",
    )
    pr_warnings: list[str] = Field(
        default_factory=list,
        description="Puerto Rico-specific warnings (e.g., missing urbanization)",
    )


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
# Enhanced Validation Request/Response Models
# ============================================================================


class EnhancedValidateRequest(BaseModel):
    """Request model for enhanced address validation with detailed scoring.

    Supports multiple input formats for the address field:
    - Raw string: "101 Main St, New York, NY 10001"
    - Minimal: "101 turkey 22222" (street + city hint + ZIP)
    - With typos: "123 Main Stret" (auto-corrected)
    - Partial: "456 Oak Ave, Los Angeles" (missing ZIP)
    """

    address: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw address string to validate. Accepts various formats including partial addresses, addresses with typos, and minimal input (street + ZIP).",
        json_schema_extra={
            "examples": [
                "101 turkey 22222",
                "123 Main Stret, New York, NY 10001",
                "456 Oak Ave Apt 2B, Los Angeles CA",
                "789 Broadway, 10003",
                "URB Las Gladiolas, 101 Calle A, San Juan PR 00906",
            ]
        },
    )
    mode: str = Field(
        default="correct",
        description="Validation mode determining how the address is processed",
        json_schema_extra={
            "enum": ["validate", "complete", "correct", "standardize"],
            "examples": ["correct"],
        },
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Optional context hints to improve validation accuracy. Useful when you have partial information from other sources.",
        json_schema_extra={
            "examples": [
                {"zip": "22222", "state": "VA"},
                {"city": "New York", "state": "NY"},
                {"zip": "00906"},
            ]
        },
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "address": "101 turkey 22222",
                    "mode": "correct",
                    "context": {}
                },
                {
                    "address": "123 Main Stret, New York",
                    "mode": "correct",
                    "context": {"state": "NY", "zip": "10001"}
                },
                {
                    "address": "456 Oak Ave Apt 2B",
                    "mode": "complete",
                    "context": {"city": "Los Angeles", "state": "CA"}
                },
            ]
        }
    )


class ComponentScoreResponse(BaseModel):
    """Score details for a single address component."""

    component: str
    confidence: str
    score: float
    original_value: str | None
    validated_value: str | None
    was_corrected: bool
    was_completed: bool
    correction_reason: str | None
    alternatives: list[str]


class ValidationIssueResponse(BaseModel):
    """Issue found during validation."""

    severity: str
    component: str | None
    message: str
    suggestion: str | None
    auto_fixable: bool


class EnhancedValidateResponse(BaseModel):
    """Comprehensive response for enhanced address validation.

    Provides detailed validation results including:
    - Overall validity and deliverability assessment
    - Component-level confidence scores
    - Automatic corrections and completions
    - USPS-standardized formatting
    - Alternative suggestions
    - Puerto Rico-specific handling
    """

    # Overall status
    is_valid: bool = Field(
        description="Whether the address passes validation (has all required components)",
        json_schema_extra={"example": True},
    )
    is_deliverable: bool = Field(
        description="Whether the address is likely deliverable by USPS (valid + high confidence)",
        json_schema_extra={"example": True},
    )
    overall_confidence: float = Field(
        description="Overall confidence score from 0.0 (no confidence) to 1.0 (fully verified)",
        json_schema_extra={"example": 0.95},
    )
    confidence_percent: float = Field(
        description="Confidence expressed as percentage (0-100) for display purposes",
        json_schema_extra={"example": 95.0},
    )
    quality: str = Field(
        description="Quality classification based on component completeness",
        json_schema_extra={
            "enum": ["complete", "partial", "ambiguous", "invalid"],
            "example": "complete",
        },
    )
    status: str = Field(
        description="Final verification status after processing",
        json_schema_extra={
            "enum": ["verified", "corrected", "completed", "suggested", "unverified", "failed"],
            "example": "corrected",
        },
    )

    # Address data
    original: dict[str, Any] = Field(
        description="Original address as parsed (before corrections)",
        json_schema_extra={
            "example": {
                "raw": "101 turkey 22222",
                "street_number": "101",
                "street_name": "Turkey",
                "city": None,
                "state": None,
                "zip_code": "22222",
            }
        },
    )
    validated: dict[str, Any] | None = Field(
        description="Validated/corrected address with all components (null if invalid)",
        json_schema_extra={
            "example": {
                "street_number": "101",
                "street_name": "Turkey Run",
                "street_type": "Rd",
                "city": "Springfield",
                "state": "VA",
                "zip_code": "22222",
            }
        },
    )
    standardized: str | None = Field(
        description="USPS-formatted single line address (all caps, standard abbreviations)",
        json_schema_extra={"example": "101 TURKEY RUN RD, SPRINGFIELD, VA 22222"},
    )

    # Component-level details
    component_scores: list[ComponentScoreResponse] = Field(
        description="Detailed confidence scores for each address component (street, city, state, zip)",
    )
    issues: list[ValidationIssueResponse] = Field(
        description="Issues found during validation (errors, warnings, info)",
    )
    corrections_applied: list[str] = Field(
        description="List of automatic corrections made (e.g., 'Fixed typo: Stret -> Street')",
        json_schema_extra={"example": ["Completed street name: Turkey -> Turkey Run"]},
    )
    completions_applied: list[str] = Field(
        description="List of missing components that were filled in",
        json_schema_extra={"example": ["Inferred city from ZIP: Springfield", "Inferred state from ZIP: VA"]},
    )

    # Alternatives
    alternatives: list[dict[str, Any]] = Field(
        description="Alternative address suggestions if primary validation uncertain",
    )

    # Puerto Rico specific
    is_puerto_rico: bool = Field(
        default=False,
        description="True if address is in Puerto Rico (ZIP codes 006-009)",
    )
    urbanization_status: str | None = Field(
        default=None,
        description="Puerto Rico urbanization (URB) status",
        json_schema_extra={
            "enum": ["present", "missing", "inferred", None],
            "example": None,
        },
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata (timing, source, etc.)",
    )


class QuickValidateResponse(BaseModel):
    """Simplified response for quick validation checks."""

    valid: bool
    deliverable: bool
    confidence: float
    confidence_percent: float
    status: str
    formatted_address: str | None
    issues_count: int
    corrections_count: int
    primary_issue: str | None


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

    # Extract PR-specific warnings from metadata
    pr_warnings: list[str] = []
    if result.original.is_puerto_rico and not result.original.urbanization:
        pr_warnings.append("Puerto Rico address missing urbanization (URB) - deliverability uncertain")
    if result.metadata.get("pr_warnings"):
        pr_warnings.extend(result.metadata["pr_warnings"])

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
        is_puerto_rico=result.original.is_puerto_rico,
        urbanization=result.original.urbanization,
        pr_warnings=pr_warnings,
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
        "orchestrator": _orchestrator is not None,
        "validator_engine": _validator_engine is not None,
        "index_ready": _pipeline.index.is_indexed if _pipeline else False,
        "completer_available": _pipeline.completer.available if _pipeline else False,
    }


# ============================================================================
# Enhanced Validation Endpoints
# ============================================================================


@router.post(
    "/validate",
    response_model=EnhancedValidateResponse,
    summary="Validate address with detailed scoring",
    description="""
## Enhanced Address Validation

Validates an address with comprehensive component-level scoring and detailed feedback.

### Supported Input Formats

| Format | Example |
|--------|---------|
| **Standard** | `123 Main St, New York, NY 10001` |
| **Minimal** | `101 turkey 22222` (street hint + ZIP) |
| **With typos** | `123 Main Stret` (auto-corrected) |
| **Partial** | `456 Oak Ave, Los Angeles` (missing ZIP) |
| **Puerto Rico** | `URB Las Gladiolas, 101 Calle A, San Juan PR 00906` |

### Validation Modes

- **validate**: Check validity only, no modifications
- **complete**: Fill in missing components (city/state from ZIP)
- **correct**: Fix typos AND complete missing parts *(default)*
- **standardize**: Format to USPS standard (all caps, abbreviations)

### Features

- ✅ Component-level confidence scores (street, city, state, zip)
- ✅ Automatic typo correction with tracking
- ✅ Missing component inference from ZIP database
- ✅ USPS-standardized formatting
- ✅ Puerto Rico urbanization (URB) handling
- ✅ Alternative suggestions when uncertain
    """,
    responses={
        200: {
            "description": "Validation successful",
            "content": {
                "application/json": {
                    "example": {
                        "is_valid": True,
                        "is_deliverable": True,
                        "overall_confidence": 0.95,
                        "confidence_percent": 95.0,
                        "quality": "complete",
                        "status": "corrected",
                        "standardized": "101 TURKEY RUN RD, SPRINGFIELD, VA 22222",
                        "corrections_applied": ["Completed street name: Turkey -> Turkey Run"],
                        "completions_applied": ["Inferred city: Springfield", "Inferred state: VA"],
                    }
                }
            },
        },
        400: {"description": "Invalid request format"},
        422: {"description": "Validation error - malformed address data"},
        503: {"description": "Validator engine not initialized"},
    },
)
async def validate_address_enhanced(
    request: EnhancedValidateRequest,
) -> EnhancedValidateResponse:
    """Validate an address with comprehensive scoring and detailed feedback."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    start_time = time.perf_counter()

    # Parse validation mode
    try:
        mode = ValidationMode(request.mode.lower())
    except ValueError:
        mode = ValidationMode.CORRECT

    # Run validation
    result = _validator_engine.validate(
        raw_address=request.address,
        mode=mode,
        context=request.context,
    )

    # Calculate processing time
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Log to audit trail
    validation_audit_log.log(ValidationAuditEntry(
        timestamp=datetime.utcnow(),
        input_address=request.address,
        validation_mode=mode.value,
        result_status=result.status.value,
        confidence=result.overall_confidence,
        is_valid=result.is_valid,
        is_deliverable=result.is_deliverable,
        corrections_count=len(result.corrections_applied),
        completions_count=len(result.completions_applied),
        processing_time_ms=processing_time_ms,
        source="validator_engine",
        error=result.metadata.get("error"),
    ))

    # Convert component scores
    component_scores = [
        ComponentScoreResponse(
            component=cs.component.value,
            confidence=cs.confidence.value,
            score=round(cs.score, 4),
            original_value=cs.original_value,
            validated_value=cs.validated_value,
            was_corrected=cs.was_corrected,
            was_completed=cs.was_completed,
            correction_reason=cs.correction_reason,
            alternatives=cs.alternatives,
        )
        for cs in result.component_scores
    ]

    # Convert issues
    issues = [
        ValidationIssueResponse(
            severity=issue.severity,
            component=issue.component.value if issue.component else None,
            message=issue.message,
            suggestion=issue.suggestion,
            auto_fixable=issue.auto_fixable,
        )
        for issue in result.issues
    ]

    return EnhancedValidateResponse(
        is_valid=result.is_valid,
        is_deliverable=result.is_deliverable,
        overall_confidence=round(result.overall_confidence, 4),
        confidence_percent=round(result.overall_confidence * 100, 1),
        quality=result.quality.value,
        status=result.status.value,
        original=result.original.to_dict(),
        validated=result.validated.to_dict() if result.validated else None,
        standardized=result.standardized,
        component_scores=component_scores,
        issues=issues,
        corrections_applied=result.corrections_applied,
        completions_applied=result.completions_applied,
        alternatives=[a.to_dict() for a in result.alternatives],
        is_puerto_rico=result.is_puerto_rico,
        urbanization_status=result.urbanization_status,
        metadata=result.metadata,
    )


@router.get(
    "/validate/quick",
    response_model=QuickValidateResponse,
    summary="Quick address validation (simplified)",
    description="""
## Quick Address Validation

Fast validation endpoint returning essential information only.
Use this for high-volume validation where detailed component scores aren't needed.

### Response Fields
- `valid`: Does the address have all required components?
- `deliverable`: Is it likely deliverable by USPS?
- `confidence`: Confidence score (0.0-1.0)
- `formatted_address`: USPS-standardized format
- `primary_issue`: Most important issue (if any)

### Performance
~2-5ms faster than `/validate` endpoint due to simplified response processing.
    """,
    responses={
        200: {
            "description": "Quick validation successful",
            "content": {
                "application/json": {
                    "example": {
                        "valid": True,
                        "deliverable": True,
                        "confidence": 0.95,
                        "confidence_percent": 95.0,
                        "status": "corrected",
                        "formatted_address": "101 TURKEY RUN RD, SPRINGFIELD, VA 22222",
                        "issues_count": 0,
                        "corrections_count": 1,
                        "primary_issue": None,
                    }
                }
            },
        },
        503: {"description": "Validator engine not initialized"},
    },
)
async def validate_address_quick(
    address: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Raw address string to validate",
        examples=["101 turkey 22222", "123 Main St, New York, NY 10001"],
    ),
    mode: str = Query(
        default="correct",
        description="Validation mode: validate, complete, correct, standardize",
        examples=["correct"],
    ),
) -> QuickValidateResponse:
    """Quick address validation with simplified response."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    try:
        validation_mode = ValidationMode(mode.lower())
    except ValueError:
        validation_mode = ValidationMode.CORRECT

    result = _validator_engine.validate(
        raw_address=address,
        mode=validation_mode,
    )

    # Get primary issue if any
    primary_issue = None
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    if errors:
        primary_issue = errors[0].message
    elif warnings:
        primary_issue = warnings[0].message

    return QuickValidateResponse(
        valid=result.is_valid,
        deliverable=result.is_deliverable,
        confidence=round(result.overall_confidence, 4),
        confidence_percent=round(result.overall_confidence * 100, 1),
        status=result.status.value,
        formatted_address=result.standardized,
        issues_count=len(result.issues),
        corrections_count=len(result.corrections_applied),
        primary_issue=primary_issue,
    )


@router.post(
    "/validate/batch",
    summary="Batch address validation",
    description="""
## Batch Address Validation

Validate multiple addresses in a single request with aggregated statistics.

### Input Formats
Each address in the list can be in any supported format:
- Standard: `"123 Main St, New York, NY 10001"`
- Minimal: `"101 turkey 22222"`
- Partial: `"456 Oak Ave, Los Angeles"`

### Response Summary
Returns individual results plus aggregate statistics:
- `valid_rate`: Percentage of valid addresses
- `deliverable_rate`: Percentage likely deliverable
- `average_confidence`: Mean confidence across all addresses

### Performance
Processes addresses sequentially. For high-volume batches (1000+),
consider using `/verify/batch` which supports concurrent processing.
    """,
    responses={
        200: {
            "description": "Batch validation successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "total": 3,
                        "valid_count": 2,
                        "deliverable_count": 2,
                        "valid_rate": 66.7,
                        "deliverable_rate": 66.7,
                        "average_confidence": 0.85,
                        "average_confidence_percent": 85.0,
                        "results": [
                            {"address": "101 turkey 22222", "is_valid": True, "confidence": 0.95},
                            {"address": "invalid address", "is_valid": False, "confidence": 0.15},
                        ],
                    }
                }
            },
        },
        503: {"description": "Validator engine not initialized"},
    },
)
async def validate_batch_enhanced(
    addresses: list[str],
    mode: str = "correct",
) -> dict[str, Any]:
    """Validate multiple addresses with enhanced scoring."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    try:
        validation_mode = ValidationMode(mode.lower())
    except ValueError:
        validation_mode = ValidationMode.CORRECT

    results = []
    valid_count = 0
    deliverable_count = 0
    total_confidence = 0.0

    for addr in addresses:
        result = _validator_engine.validate(addr, validation_mode)
        results.append({
            "address": addr,
            "is_valid": result.is_valid,
            "is_deliverable": result.is_deliverable,
            "confidence": round(result.overall_confidence, 4),
            "confidence_percent": round(result.overall_confidence * 100, 1),
            "status": result.status.value,
            "standardized": result.standardized,
            "corrections_count": len(result.corrections_applied),
            "issues_count": len(result.issues),
        })
        if result.is_valid:
            valid_count += 1
        if result.is_deliverable:
            deliverable_count += 1
        total_confidence += result.overall_confidence

    avg_confidence = total_confidence / len(addresses) if addresses else 0

    return {
        "success": True,
        "total": len(addresses),
        "valid_count": valid_count,
        "deliverable_count": deliverable_count,
        "valid_rate": round(valid_count / len(addresses) * 100, 1) if addresses else 0,
        "deliverable_rate": round(deliverable_count / len(addresses) * 100, 1) if addresses else 0,
        "average_confidence": round(avg_confidence, 4),
        "average_confidence_percent": round(avg_confidence * 100, 1),
        "results": results,
    }


@router.post(
    "/complete",
    summary="Complete partial address",
    description="""
## Address Completion

Fills in missing components of a partial address using ZIP code inference
and known address databases.

### Completion Capabilities
- **City from ZIP**: `"101 Main St, 10001"` → `"101 Main St, New York, NY 10001"`
- **State from ZIP**: `"123 Oak Ave, Boston"` → `"123 Oak Ave, Boston, MA"`
- **Street type**: `"456 Maple, Chicago IL"` → `"456 Maple St, Chicago, IL"`
- **Puerto Rico URB**: Infers urbanization when possible

### When to Use
- User submitted partial form data
- Importing addresses from legacy systems
- Autocomplete/typeahead functionality

### Context Hints
Provide optional context to improve completion accuracy:
```json
{"zip": "10001", "state": "NY"}
```
    """,
    responses={
        200: {
            "description": "Completion successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "standardized": "101 MAIN ST, NEW YORK, NY 10001",
                        "completions_made": ["Inferred city: New York", "Inferred state: NY"],
                        "confidence": 0.92,
                        "is_complete": True,
                    }
                }
            },
        },
        503: {"description": "Validator engine not initialized"},
    },
)
async def complete_address(
    address: str,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Complete a partial address with missing components."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    result = _validator_engine.validate(
        raw_address=address,
        mode=ValidationMode.COMPLETE,
        context=context or {},
    )

    return {
        "success": result.is_valid,
        "original": result.original.to_dict(),
        "completed": result.validated.to_dict() if result.validated else None,
        "standardized": result.standardized,
        "completions_made": result.completions_applied,
        "confidence": round(result.overall_confidence, 4),
        "confidence_percent": round(result.overall_confidence * 100, 1),
        "is_complete": len([
            cs for cs in result.component_scores
            if cs.confidence.value == "missing"
        ]) == 0,
    }


@router.post(
    "/correct",
    summary="Correct address errors",
    description="""
## Address Correction

Automatically fixes common errors in addresses while preserving intent.

### Correction Capabilities

| Error Type | Example |
|------------|---------|
| **Street typos** | `Stret` → `Street`, `Avenu` → `Avenue` |
| **State misspellings** | `Calfornia` → `California` |
| **Abbreviation expansion** | `St` → `Street` (when appropriate) |
| **Case normalization** | `NEW YORK` → `New York` |
| **Punctuation cleanup** | Extra commas, periods removed |

### Correction + Completion
This endpoint also completes missing components after corrections.
Use `mode=validate` in `/validate` if you only want error checking.

### Audit Trail
Returns `corrections_made` array showing all changes applied.
    """,
    responses={
        200: {
            "description": "Correction successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "standardized": "123 MAIN STREET, NEW YORK, NY 10001",
                        "corrections_made": ["Fixed typo: Stret -> Street", "Fixed state: NY"],
                        "confidence": 0.94,
                        "was_corrected": True,
                    }
                }
            },
        },
        503: {"description": "Validator engine not initialized"},
    },
)
async def correct_address(
    address: str,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Correct errors in an address."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    result = _validator_engine.validate(
        raw_address=address,
        mode=ValidationMode.CORRECT,
        context=context or {},
    )

    return {
        "success": result.is_valid,
        "original": result.original.to_dict(),
        "corrected": result.validated.to_dict() if result.validated else None,
        "standardized": result.standardized,
        "corrections_made": result.corrections_applied,
        "confidence": round(result.overall_confidence, 4),
        "confidence_percent": round(result.overall_confidence * 100, 1),
        "was_corrected": len(result.corrections_applied) > 0,
    }


@router.post(
    "/standardize",
    summary="Format to USPS standard",
    description="""
## USPS Address Standardization

Converts an address to official USPS standard format for mailing and database consistency.

### USPS Format Rules
- **ALL CAPS**: `123 MAIN ST`
- **Standard abbreviations**: `Street` → `ST`, `Avenue` → `AVE`
- **No punctuation**: Commas and periods removed from address line
- **Directional abbreviations**: `North` → `N`, `Southwest` → `SW`
- **Unit designator format**: `APT 2B`, `STE 100`

### Puerto Rico Handling
Puerto Rico addresses follow special USPS rules:
```
URB LAS GLADIOLAS
101 CALLE A
SAN JUAN PR 00906-1234
```

### Example Transformation
**Input**: `123 north main street, apartment 2b, new york, new york 10001`
**Output**: `123 N MAIN ST APT 2B, NEW YORK, NY 10001`
    """,
    responses={
        200: {
            "description": "Standardization successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "original": "123 Main Street, New York, NY 10001",
                        "standardized": "123 MAIN ST, NEW YORK, NY 10001",
                        "confidence": 0.98,
                        "is_puerto_rico": False,
                    }
                }
            },
        },
        503: {"description": "Validator engine not initialized"},
    },
)
async def standardize_address(
    address: str,
) -> dict[str, Any]:
    """Format an address to USPS standard format."""
    if not _validator_engine:
        raise HTTPException(status_code=503, detail="Validator engine not initialized")

    result = _validator_engine.validate(
        raw_address=address,
        mode=ValidationMode.STANDARDIZE,
    )

    return {
        "success": result.is_valid,
        "original": address,
        "standardized": result.standardized,
        "confidence": round(result.overall_confidence, 4),
        "is_puerto_rico": result.is_puerto_rico,
    }


# ============================================================================
# Agent-Based Verification (5-Agent Architecture)
# ============================================================================


class AgentVerifyRequest(BaseModel):
    """Request model for agent-based address verification."""

    address: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw address string to verify",
        examples=["101 turkey", "123 Main St", "456 Oak Ave Las Vegas"],
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation context",
    )
    session_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Conversation history for context extraction",
    )
    hints: dict[str, str] | None = Field(
        default=None,
        description="Explicit hints like {'state': 'NV', 'city': 'Las Vegas'}",
    )
    multi_state: bool = Field(
        default=True,
        description="Enable multi-state results when state uncertain",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum matches to return",
    )
    include_trace: bool = Field(
        default=True,
        description="Include visual pipeline trace in response",
    )


class AgentVerifyResponse(BaseModel):
    """Response model for agent-based verification."""

    success: bool
    status: str
    verified: dict[str, Any] | None
    confidence: float
    alternatives: list[dict[str, Any]]
    multi_state_results: dict[str, list[dict[str, Any]]] | None
    inferences_made: dict[str, Any]
    context_used: dict[str, Any]
    quality_gates: list[dict[str, Any]]
    processing_time_ms: int
    agent_timings: dict[str, int]
    pipeline_trace: dict[str, Any] | None = Field(
        default=None,
        description="Visual pipeline trace showing query flow through agents",
    )


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics."""

    agents: dict[str, Any]
    indexes: dict[str, Any]


@router.post("/verify/agent", response_model=AgentVerifyResponse)
async def verify_with_agents(request: AgentVerifyRequest) -> AgentVerifyResponse:
    """Verify address using 5-agent intelligent pipeline.

    This endpoint uses the new agent-based architecture for:
    - Context extraction from conversation history
    - Intelligent inference of missing components
    - Multi-state matching when state is uncertain
    - Semantic vector search for similar addresses
    - Quality gate validation

    The 5 agents are:
    1. Context Agent - Extracts geographic context from history
    2. Parser Agent - Normalizes and parses the address
    3. Inference Agent - Infers missing state, city, ZIP
    4. Match Agent - Finds matches using fuzzy + semantic search
    5. Enforcer Agent - Validates results with quality gates

    Args:
        request: Agent verification request with optional context.

    Returns:
        Comprehensive verification result with inferences, multi-state results, and pipeline trace.
    """
    if not _orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Agent orchestrator not initialized. Use /verify for legacy pipeline."
        )

    result, trace = await _orchestrator.process(
        raw_address=request.address,
        session_id=request.session_id,
        session_history=request.session_history,
        hints=request.hints,
        max_results=request.max_results,
        enable_trace=request.include_trace,
    )

    # Convert ParsedAddress objects to dicts
    verified_dict = None
    if result.verified_address:
        verified_dict = {
            "street_number": result.verified_address.street_number,
            "street_name": result.verified_address.street_name,
            "street_type": result.verified_address.street_type,
            "city": result.verified_address.city,
            "state": result.verified_address.state,
            "zip_code": result.verified_address.zip_code,
            "single_line": result.verified_address.single_line,
        }

    alternatives_list = []
    for alt in result.alternatives:
        alternatives_list.append({
            "street_number": alt.street_number,
            "street_name": alt.street_name,
            "street_type": alt.street_type,
            "city": alt.city,
            "state": alt.state,
            "zip_code": alt.zip_code,
            "single_line": alt.single_line,
        })

    # Convert multi-state results
    multi_state_dict = None
    if result.multi_state_results:
        multi_state_dict = {}
        for state, addresses in result.multi_state_results.items():
            multi_state_dict[state] = [
                {
                    "street_number": a.street_number,
                    "street_name": a.street_name,
                    "street_type": a.street_type,
                    "city": a.city,
                    "state": a.state,
                    "zip_code": a.zip_code,
                    "single_line": a.single_line,
                }
                for a in addresses
            ]

    # Convert quality gates
    quality_gates_list = [
        {
            "gate": g.gate.value,
            "passed": g.passed,
            "message": g.message,
            "details": g.details,
        }
        for g in result.quality_gates
    ]

    return AgentVerifyResponse(
        success=result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.CORRECTED,
            VerificationStatus.COMPLETED,
        ),
        status=result.status.value,
        verified=verified_dict,
        confidence=result.confidence,
        alternatives=alternatives_list,
        multi_state_results=multi_state_dict,
        inferences_made=result.metadata.get("inferences_made", {}),
        context_used={
            "confidence": result.metadata.get("context_confidence", 0),
            "signals": result.metadata.get("context_signals", []),
        },
        quality_gates=quality_gates_list,
        processing_time_ms=result.metadata.get("total_time_ms", 0),
        agent_timings=result.metadata.get("agent_timings", {}),
        pipeline_trace=trace.to_dict() if trace else None,
    )


@router.get("/agents/stats", response_model=AgentStatsResponse)
async def get_agent_stats() -> AgentStatsResponse:
    """Get statistics for all agents in the pipeline.

    Returns:
        Agent performance statistics and index information.
    """
    if not _orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Agent orchestrator not initialized"
        )

    stats = _orchestrator.get_agent_stats()

    return AgentStatsResponse(
        agents={
            "context": stats.get("context_agent", {}),
            "parser": stats.get("parser_agent", {}),
            "inference": stats.get("inference_agent", {}),
            "match": stats.get("match_agent", {}),
            "enforcer": stats.get("enforcer_agent", {}),
        },
        indexes=stats.get("indexes", {}),
    )


# ============================================================================
# Audit & Monitoring Endpoints
# ============================================================================


@router.get(
    "/audit/stats",
    summary="Get validation audit statistics",
    description="Returns aggregate statistics from the validation audit log.",
)
async def get_audit_stats() -> dict[str, Any]:
    """Get validation audit statistics."""
    return validation_audit_log.get_stats()


@router.get(
    "/audit/recent",
    summary="Get recent validation entries",
    description="Returns the most recent validation audit entries.",
)
async def get_audit_recent(
    count: int = Query(default=10, ge=1, le=100, description="Number of entries"),
) -> list[dict[str, Any]]:
    """Get recent validation audit entries."""
    return validation_audit_log.get_recent(count)


@router.get(
    "/audit/errors",
    summary="Get recent validation errors",
    description="Returns recent validation entries that had errors.",
)
async def get_audit_errors(
    count: int = Query(default=10, ge=1, le=100, description="Number of entries"),
) -> list[dict[str, Any]]:
    """Get recent validation error entries."""
    return validation_audit_log.get_errors(count)
