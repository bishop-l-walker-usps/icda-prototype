"""API router for address completion pipeline.

Provides endpoints for:
- Single address completion
- Batch address completion
- Pipeline statistics
- Index management
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/completion", tags=["Address Completion"])

# Module-level references (set by configure_completion_router)
_pipeline = None
_vector_index = None
_embedder = None
_db = None


def configure_completion_router(
    pipeline,
    vector_index=None,
    embedder=None,
    db=None
):
    """Configure router with pipeline dependencies.

    Args:
        pipeline: AddressCompletionPipeline instance
        vector_index: RedisAddressIndex instance (optional)
        embedder: AddressEmbedder instance (optional)
        db: CustomerDB instance for indexing (optional)
    """
    global _pipeline, _vector_index, _embedder, _db
    _pipeline = pipeline
    _vector_index = vector_index
    _embedder = embedder
    _db = db


# Request/Response Models

class CompletionRequest(BaseModel):
    """Request for address completion."""
    address: str = Field(..., min_length=1, max_length=500, description="Partial or incomplete address")
    use_cache: bool = Field(True, description="Use cached results if available")
    return_suggestions: bool = Field(False, description="Include alternative suggestions")


class BatchCompletionRequest(BaseModel):
    """Request for batch address completion."""
    addresses: List[str] = Field(..., min_items=1, max_items=1000, description="List of addresses to complete")
    concurrency: int = Field(10, ge=1, le=50, description="Max concurrent completions")


class CompletionResponse(BaseModel):
    """Response for address completion."""
    success: bool
    original: str
    completed: Optional[str]
    confidence: float
    source: str
    crid: Optional[str] = None
    suggestions: Optional[List[str]] = None
    reason: Optional[str] = None


class BatchCompletionResponse(BaseModel):
    """Response for batch completion."""
    success: bool
    total: int
    completed: int
    failed: int
    from_cache: int
    from_vector: int
    from_nova: int
    results: List[CompletionResponse]


class IndexBuildRequest(BaseModel):
    """Request to build/rebuild index."""
    recreate: bool = Field(False, description="Drop and recreate index")
    warmup_cache: bool = Field(True, description="Pre-populate cache with common patterns")


class IndexBuildResponse(BaseModel):
    """Response from index build."""
    success: bool
    indexed: int
    failed: int
    skipped: int
    cache_warmed: Optional[int] = None
    error: Optional[str] = None


# Endpoints

@router.post("/complete", response_model=CompletionResponse)
async def complete_address(request: CompletionRequest):
    """Complete a partial or incomplete address.

    Uses the three-tier pipeline:
    1. L1 Cache: Check for cached completions
    2. L2 Vector Search: Find similar addresses using Titan embeddings
    3. L3 Nova Rerank: Use Nova LLM to select best match

    Returns the completed address with confidence score and source.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Completion pipeline not initialized")

    try:
        result = await _pipeline.complete(
            address=request.address,
            use_cache=request.use_cache,
            return_suggestions=request.return_suggestions
        )

        return CompletionResponse(
            success=result.completed is not None,
            original=result.original,
            completed=result.completed,
            confidence=result.confidence,
            source=result.source.value,
            crid=result.crid,
            suggestions=result.suggestions,
            reason=result.reason
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/batch", response_model=BatchCompletionResponse)
async def complete_batch(request: BatchCompletionRequest):
    """Complete multiple addresses in parallel.

    Efficiently processes a batch of addresses using concurrent requests.
    Returns results for each address along with aggregate statistics.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Completion pipeline not initialized")

    try:
        batch_result = await _pipeline.complete_batch(
            addresses=request.addresses,
            concurrency=request.concurrency
        )

        return BatchCompletionResponse(
            success=True,
            total=batch_result.total,
            completed=batch_result.completed,
            failed=batch_result.failed,
            from_cache=batch_result.from_cache,
            from_vector=batch_result.from_vector,
            from_nova=batch_result.from_nova,
            results=[
                CompletionResponse(
                    success=r.completed is not None,
                    original=r.original,
                    completed=r.completed,
                    confidence=r.confidence,
                    source=r.source.value,
                    crid=r.crid,
                    reason=r.reason
                )
                for r in batch_result.results
            ]
        )

    except Exception as e:
        logger.error(f"Batch completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get completion pipeline statistics.

    Returns information about:
    - Pipeline availability
    - Embedder status
    - Vector index status
    - Cache statistics
    """
    if not _pipeline:
        return {
            "success": False,
            "available": False,
            "error": "Pipeline not initialized"
        }

    try:
        stats = await _pipeline.get_stats()
        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/index/build", response_model=IndexBuildResponse)
async def build_index(request: IndexBuildRequest):
    """Build or rebuild the address vector index.

    Indexes all customer addresses into Redis for vector similarity search.
    Optionally warms the cache with common address patterns.
    """
    if not _vector_index:
        raise HTTPException(status_code=503, detail="Vector index not available")
    if not _db:
        raise HTTPException(status_code=503, detail="Customer database not available")

    try:
        # Create/recreate index
        if request.recreate:
            await _vector_index.delete_all()

        created = await _vector_index.create_index(recreate=request.recreate)
        if not created:
            return IndexBuildResponse(
                success=False,
                indexed=0,
                failed=0,
                skipped=0,
                error="Failed to create index"
            )

        # Index customers
        stats = await _vector_index.index_customers_batch(
            customers=_db.customers,
            batch_size=100
        )

        # Warm cache
        cache_warmed = None
        if request.warmup_cache and _pipeline:
            warmup_stats = await _pipeline.warmup_cache(_db.customers)
            cache_warmed = warmup_stats.get("exact", 0) + warmup_stats.get("variations", 0)

        return IndexBuildResponse(
            success=True,
            indexed=stats.get("indexed", 0),
            failed=stats.get("failed", 0),
            skipped=stats.get("skipped", 0),
            cache_warmed=cache_warmed
        )

    except Exception as e:
        logger.error(f"Index build error: {e}")
        return IndexBuildResponse(
            success=False,
            indexed=0,
            failed=0,
            skipped=0,
            error=str(e)
        )


@router.delete("/cache")
async def clear_cache():
    """Clear the completion cache.

    Removes all cached completion results, forcing fresh lookups.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        count = await _pipeline.clear_cache()
        return {
            "success": True,
            "cleared": count
        }

    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for completion pipeline."""
    return {
        "available": _pipeline is not None and _pipeline.available,
        "vector_index": _vector_index is not None and _vector_index.available if _vector_index else False,
        "embedder": _embedder is not None and _embedder.available if _embedder else False,
    }
