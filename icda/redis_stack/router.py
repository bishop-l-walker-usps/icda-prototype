"""Redis Stack API endpoints for analytics, audit, and real-time features.

Provides:
- /api/analytics/* - Query analytics and metrics
- /api/audit/* - Query audit trail and event streams
- /api/suggest/* - Autocomplete suggestions
- /api/events/* - Real-time event streaming (SSE)
- /api/redis-stack/* - Module status and health
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Redis Stack"])

# Module references (set by configure_router)
_redis_stack = None
_db = None


def configure_router(redis_stack, db=None):
    """Configure router with module instances."""
    global _redis_stack, _db
    _redis_stack = redis_stack
    _db = db


# =============================================================================
# Redis Stack Status
# =============================================================================

@router.get("/api/redis-stack/status")
async def redis_stack_status():
    """Get Redis Stack module availability and health."""
    if not _redis_stack:
        return {"available": False, "error": "Redis Stack not initialized"}

    health = await _redis_stack.health_check()
    return {
        "available": _redis_stack._connected,
        "modules": _redis_stack._get_module_status(),
        "health": health,
    }


@router.get("/api/redis-stack/modules")
async def redis_stack_modules():
    """Get detailed info on available Redis Stack modules."""
    if not _redis_stack:
        return {"available": False}

    result = {
        "connected": _redis_stack._connected,
        "modules": {},
    }

    # Get per-module stats if available
    if _redis_stack.timeseries_available and _redis_stack.timeseries:
        try:
            ts_health = await _redis_stack.timeseries.health_check()
            result["modules"]["timeseries"] = ts_health
        except Exception as e:
            result["modules"]["timeseries"] = {"error": str(e)}

    if _redis_stack.bloom_available and _redis_stack.bloom:
        try:
            bloom_stats = await _redis_stack.bloom.get_filter_stats()
            result["modules"]["bloom"] = bloom_stats
        except Exception as e:
            result["modules"]["bloom"] = {"error": str(e)}

    if _redis_stack.streams and _redis_stack.streams:
        try:
            streams_stats = await _redis_stack.streams.get_all_stats()
            result["modules"]["streams"] = streams_stats
        except Exception as e:
            result["modules"]["streams"] = {"error": str(e)}

    if _redis_stack.search_available and _redis_stack.search:
        try:
            search_info = await _redis_stack.search.get_index_info("idx:customers")
            result["modules"]["search"] = search_info
        except Exception as e:
            result["modules"]["search"] = {"error": str(e)}

    return result


# =============================================================================
# Analytics Endpoints (RedisTimeSeries)
# =============================================================================

@router.get("/api/analytics/queries")
async def analytics_queries(
    range: str = Query("1h", description="Time range: 1h, 24h, 7d")
):
    """Get query analytics for time range."""
    if not _redis_stack or not _redis_stack.timeseries_available:
        return {"available": False, "error": "RedisTimeSeries not available"}

    range_ms = {
        "1h": 3600 * 1000,
        "24h": 86400 * 1000,
        "7d": 604800 * 1000,
    }.get(range, 3600 * 1000)

    stats = await _redis_stack.timeseries.get_query_stats(range_ms)
    return {
        "range": range,
        "stats": stats,
    }


@router.get("/api/analytics/agents")
async def analytics_agents(
    range: str = Query("1h", description="Time range: 1h, 24h, 7d")
):
    """Get per-agent performance metrics."""
    if not _redis_stack or not _redis_stack.timeseries_available:
        return {"available": False, "error": "RedisTimeSeries not available"}

    range_ms = {
        "1h": 3600 * 1000,
        "24h": 86400 * 1000,
        "7d": 604800 * 1000,
    }.get(range, 3600 * 1000)

    stats = await _redis_stack.timeseries.get_agent_stats(range_ms)
    return {
        "range": range,
        "agents": stats,
    }


@router.get("/api/analytics/timeseries/{key}")
async def analytics_timeseries(
    key: str,
    range: str = Query("1h"),
    bucket: str = Query("1m", description="Aggregation bucket: 1m, 5m, 1h"),
):
    """Get raw time series data for graphing."""
    if not _redis_stack or not _redis_stack.timeseries_available:
        return {"available": False, "error": "RedisTimeSeries not available"}

    range_ms = {
        "1h": 3600 * 1000,
        "24h": 86400 * 1000,
        "7d": 604800 * 1000,
    }.get(range, 3600 * 1000)

    bucket_ms = {
        "1m": 60 * 1000,
        "5m": 300 * 1000,
        "1h": 3600 * 1000,
    }.get(bucket, 60 * 1000)

    # Prefix key if not already prefixed
    if not key.startswith("ts:"):
        key = f"ts:{key}"

    data = await _redis_stack.timeseries.get_time_series(key, range_ms, bucket_ms)
    return {
        "key": key,
        "range": range,
        "bucket": bucket,
        "points": len(data),
        "data": data,
    }


@router.get("/api/analytics/trending")
async def analytics_trending(
    k: int = Query(10, ge=1, le=100)
):
    """Get top K trending queries."""
    if not _redis_stack or not _redis_stack.bloom_available:
        return {"available": False, "error": "RedisBloom not available"}

    queries = await _redis_stack.get_trending_queries(k)
    return {
        "count": len(queries),
        "queries": [{"query": q, "count": c} for q, c in queries],
    }


# =============================================================================
# Audit Endpoints (Redis Streams)
# =============================================================================

@router.get("/api/audit/queries")
async def audit_queries(
    limit: int = Query(100, ge=1, le=1000),
    from_id: Optional[str] = None,
):
    """Get query audit trail."""
    if not _redis_stack or not _redis_stack.streams:
        return {"available": False, "error": "Redis Streams not available"}

    events = await _redis_stack.streams.get_query_audit(limit, from_id)
    return {
        "count": len(events),
        "events": events,
    }


@router.get("/api/audit/customers")
async def audit_customers(
    limit: int = Query(100, ge=1, le=1000),
    crid: Optional[str] = None,
):
    """Get customer change audit trail."""
    if not _redis_stack or not _redis_stack.streams:
        return {"available": False, "error": "Redis Streams not available"}

    events = await _redis_stack.streams.get_customer_changes(limit, crid)
    return {
        "count": len(events),
        "crid_filter": crid,
        "events": events,
    }


@router.get("/api/audit/trace/{trace_id}")
async def audit_trace(trace_id: str):
    """Get all events for a specific trace ID."""
    if not _redis_stack or not _redis_stack.streams:
        return {"available": False, "error": "Redis Streams not available"}

    events = await _redis_stack.streams.get_agent_trace(trace_id)
    return {
        "trace_id": trace_id,
        "count": len(events),
        "events": events,
    }


@router.get("/api/audit/errors")
async def audit_errors(
    limit: int = Query(100, ge=1, le=1000),
    error_type: Optional[str] = None,
):
    """Get recent errors from error stream."""
    if not _redis_stack or not _redis_stack.streams:
        return {"available": False, "error": "Redis Streams not available"}

    events = await _redis_stack.streams.get_errors(limit, error_type)
    return {
        "count": len(events),
        "error_type_filter": error_type,
        "events": events,
    }


@router.get("/api/audit/stats")
async def audit_stats():
    """Get audit stream statistics."""
    if not _redis_stack or not _redis_stack.streams:
        return {"available": False, "error": "Redis Streams not available"}

    stats = await _redis_stack.streams.get_all_stats()
    return stats


# =============================================================================
# Autocomplete Suggestions (RediSearch)
# =============================================================================

@router.get("/api/suggest/address")
async def suggest_address(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    fuzzy: bool = Query(True),
):
    """Get address autocomplete suggestions."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    suggestions = await _redis_stack.search.suggest_address(q, limit, fuzzy)
    return {
        "query": q,
        "count": len(suggestions),
        "suggestions": [{"text": s.text, "score": s.score} for s in suggestions],
    }


@router.get("/api/suggest/city")
async def suggest_city(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    fuzzy: bool = Query(True),
):
    """Get city autocomplete suggestions."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    suggestions = await _redis_stack.search.suggest_city(q, limit, fuzzy)
    return {
        "query": q,
        "count": len(suggestions),
        "suggestions": [{"text": s.text, "score": s.score, "state": s.payload} for s in suggestions],
    }


@router.get("/api/suggest/name")
async def suggest_name(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    fuzzy: bool = Query(True),
):
    """Get name autocomplete suggestions."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    suggestions = await _redis_stack.search.suggest_name(q, limit, fuzzy)
    return {
        "query": q,
        "count": len(suggestions),
        "suggestions": [{"text": s.text, "score": s.score} for s in suggestions],
    }


@router.post("/api/suggest/build")
async def build_suggestions():
    """Build suggestion dictionaries from customer data."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"success": False, "error": "RediSearch not available"}

    if not _db:
        return {"success": False, "error": "Customer database not available"}

    stats = await _redis_stack.search.build_suggestions_from_customers(_db.customers)
    return {
        "success": True,
        "stats": stats,
    }


# =============================================================================
# Faceted Search (RediSearch)
# =============================================================================

@router.get("/api/facets/state")
async def facets_state(
    q: str = Query("*", description="Search query (use * for all)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get customer counts by state."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    result = await _redis_stack.search.get_state_facets(q, limit)
    return {
        "field": result.field,
        "count": len(result.values),
        "facets": [{"value": v, "count": c} for v, c in result.values],
    }


@router.get("/api/facets/city")
async def facets_city(
    q: str = Query("*"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get customer counts by city."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    result = await _redis_stack.search.get_city_facets(q, limit)
    return {
        "field": result.field,
        "count": len(result.values),
        "facets": [{"value": v, "count": c} for v, c in result.values],
    }


@router.get("/api/facets/type")
async def facets_type(q: str = Query("*")):
    """Get customer counts by type."""
    if not _redis_stack or not _redis_stack.search_available:
        return {"available": False, "error": "RediSearch not available"}

    result = await _redis_stack.search.get_type_facets(q)
    return {
        "field": result.field,
        "count": len(result.values),
        "facets": [{"value": v, "count": c} for v, c in result.values],
    }


# =============================================================================
# Real-time Events (SSE via Pub/Sub)
# =============================================================================

@router.get("/api/events/stream")
async def events_stream(request: Request):
    """Server-Sent Events stream for real-time updates.

    Subscribe to: index progress, health status, admin alerts.
    """
    if not _redis_stack or not _redis_stack.pubsub:
        return JSONResponse(
            {"error": "Redis Pub/Sub not available"},
            status_code=503,
        )

    from icda.redis_stack.redis_pubsub import RedisPubSubManager

    channels = [
        RedisPubSubManager.CHANNEL_INDEX_PROGRESS,
        RedisPubSubManager.CHANNEL_HEALTH,
        RedisPubSubManager.CHANNEL_ADMIN,
    ]

    async def event_generator():
        async for event in _redis_stack.pubsub.sse_stream(channels):
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/events/index-progress")
async def events_index_progress(request: Request):
    """SSE stream for indexing progress only."""
    if not _redis_stack or not _redis_stack.pubsub:
        return JSONResponse(
            {"error": "Redis Pub/Sub not available"},
            status_code=503,
        )

    from icda.redis_stack.redis_pubsub import RedisPubSubManager

    async def event_generator():
        async for event in _redis_stack.pubsub.sse_stream(
            [RedisPubSubManager.CHANNEL_INDEX_PROGRESS],
            heartbeat_interval=5.0,
        ):
            if await request.is_disconnected():
                break
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# =============================================================================
# Bloom Filter Operations
# =============================================================================

@router.get("/api/bloom/stats")
async def bloom_stats():
    """Get Bloom filter statistics."""
    if not _redis_stack or not _redis_stack.bloom_available:
        return {"available": False, "error": "RedisBloom not available"}

    stats = await _redis_stack.bloom.get_filter_stats()
    return stats


@router.get("/api/bloom/query-seen")
async def bloom_query_seen(q: str = Query(...)):
    """Check if query was recently seen."""
    if not _redis_stack or not _redis_stack.bloom_available:
        return {"available": False, "error": "RedisBloom not available"}

    seen = await _redis_stack.check_query_seen(q)
    return {"query": q, "seen": seen}


@router.get("/api/bloom/unique-users")
async def bloom_unique_users():
    """Get approximate unique user count (HyperLogLog)."""
    if not _redis_stack or not _redis_stack.bloom_available:
        return {"available": False, "error": "RedisBloom not available"}

    count = await _redis_stack.bloom.get_unique_count()
    return {"unique_users": count}
