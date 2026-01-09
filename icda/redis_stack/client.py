"""Unified Redis Stack client with module detection and graceful degradation.

Provides a single entry point for all Redis Stack functionality:
- Automatic module detection on connect
- Graceful degradation when modules unavailable
- Timeout protection on all operations
- Health monitoring
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .redis_search import RedisSearchEnhanced
    from .redis_json import RedisJSONWrapper
    from .redis_timeseries import RedisTimeSeriesWrapper
    from .redis_bloom import RedisBloomWrapper
    from .redis_pubsub import RedisPubSubManager
    from .redis_streams import RedisStreamsManager

logger = logging.getLogger(__name__)


class RedisStackClient:
    """Unified client for all Redis Stack modules.

    Provides:
    - Module detection and availability tracking
    - Graceful degradation when modules unavailable
    - Centralized connection management
    - Health check aggregation

    Usage:
        client = RedisStackClient()
        modules = await client.connect("redis://localhost:6379")
        print(modules)  # {'search': True, 'json': True, ...}

        if client.search_available:
            results = await client.search.suggest("addr", "123 Main")
    """

    __slots__ = (
        "redis", "url",
        # Module availability
        "search_available", "json_available", "timeseries_available",
        "bloom_available", "graph_available",
        # Module wrappers
        "search", "json", "timeseries", "bloom", "pubsub", "streams",
        # State
        "_connected", "_modules_detected",
    )

    # Module name mapping from MODULE LIST
    MODULE_NAMES = {
        "search": ["search", "ft"],
        "json": ["rejson", "json"],
        "timeseries": ["timeseries", "ts"],
        "bloom": ["bf", "bloom"],
        "graph": ["graph"],
    }

    def __init__(self):
        self.redis = None
        self.url = ""

        # Module availability flags
        self.search_available = False
        self.json_available = False
        self.timeseries_available = False
        self.bloom_available = False
        self.graph_available = False

        # Module wrappers (lazy initialized)
        self.search: "RedisSearchEnhanced | None" = None
        self.json: "RedisJSONWrapper | None" = None
        self.timeseries: "RedisTimeSeriesWrapper | None" = None
        self.bloom: "RedisBloomWrapper | None" = None
        self.pubsub: "RedisPubSubManager | None" = None
        self.streams: "RedisStreamsManager | None" = None

        self._connected = False
        self._modules_detected = {}

    async def connect(self, url: str, timeout: float = 10.0) -> dict[str, bool]:
        """Connect to Redis and detect available modules.

        Args:
            url: Redis connection URL (redis://localhost:6379)
            timeout: Connection timeout in seconds

        Returns:
            Dict of module availability: {'search': True, 'json': False, ...}
        """
        if not url:
            logger.info("RedisStack: No URL provided, all modules unavailable")
            return self._get_module_status()

        self.url = url

        try:
            import redis.asyncio as aioredis

            self.redis = aioredis.from_url(
                url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                health_check_interval=30,
            )

            # Test connection
            await asyncio.wait_for(self.redis.ping(), timeout=timeout)
            self._connected = True
            logger.info(f"RedisStack: Connected to {url}")

            # Detect modules
            await self._detect_modules(timeout)

            # Initialize available module wrappers
            await self._init_modules()

            return self._get_module_status()

        except asyncio.TimeoutError:
            logger.warning(f"RedisStack: Connection timed out after {timeout}s")
            return self._get_module_status()
        except ImportError:
            logger.warning("RedisStack: redis package not installed")
            return self._get_module_status()
        except Exception as e:
            logger.warning(f"RedisStack: Connection failed - {e}")
            return self._get_module_status()

    async def _detect_modules(self, timeout: float = 5.0) -> None:
        """Detect which Redis Stack modules are available."""
        if not self.redis:
            return

        try:
            modules = await asyncio.wait_for(
                self.redis.module_list(),
                timeout=timeout
            )

            module_names = [m.get("name", "").lower() for m in modules]
            self._modules_detected = {name: True for name in module_names}

            # Check each module type
            for module_type, names in self.MODULE_NAMES.items():
                available = any(name in module_names for name in names)
                setattr(self, f"{module_type}_available", available)

            logger.info(f"RedisStack modules detected: {self._modules_detected}")
            logger.info(f"  RediSearch: {self.search_available}")
            logger.info(f"  RedisJSON: {self.json_available}")
            logger.info(f"  RedisTimeSeries: {self.timeseries_available}")
            logger.info(f"  RedisBloom: {self.bloom_available}")

        except asyncio.TimeoutError:
            logger.warning(f"RedisStack: Module detection timed out")
        except Exception as e:
            logger.warning(f"RedisStack: Module detection failed - {e}")

    async def _init_modules(self) -> None:
        """Initialize available module wrappers."""
        # Always initialize pub/sub and streams (core Redis features)
        if self._connected:
            try:
                from .redis_pubsub import RedisPubSubManager
                self.pubsub = RedisPubSubManager(self.redis)
                logger.info("RedisStack: Pub/Sub manager initialized")
            except ImportError:
                pass

            try:
                from .redis_streams import RedisStreamsManager
                self.streams = RedisStreamsManager(self.redis)
                await self.streams.ensure_streams()
                logger.info("RedisStack: Streams manager initialized")
            except ImportError:
                pass

        # Initialize module-specific wrappers
        if self.search_available:
            try:
                from .redis_search import RedisSearchEnhanced
                self.search = RedisSearchEnhanced(self.redis)
                logger.info("RedisStack: RediSearch wrapper initialized")
            except ImportError:
                pass

        if self.json_available:
            try:
                from .redis_json import RedisJSONWrapper
                self.json = RedisJSONWrapper(self.redis)
                logger.info("RedisStack: RedisJSON wrapper initialized")
            except ImportError:
                pass

        if self.timeseries_available:
            try:
                from .redis_timeseries import RedisTimeSeriesWrapper
                self.timeseries = RedisTimeSeriesWrapper(self.redis)
                await self.timeseries.ensure_timeseries()
                logger.info("RedisStack: RedisTimeSeries wrapper initialized")
            except ImportError:
                pass

        if self.bloom_available:
            try:
                from .redis_bloom import RedisBloomWrapper
                self.bloom = RedisBloomWrapper(self.redis)
                await self.bloom.ensure_filters()
                logger.info("RedisStack: RedisBloom wrapper initialized")
            except ImportError:
                pass

    def _get_module_status(self) -> dict[str, bool]:
        """Get current module availability status."""
        return {
            "connected": self._connected,
            "search": self.search_available,
            "json": self.json_available,
            "timeseries": self.timeseries_available,
            "bloom": self.bloom_available,
            "graph": self.graph_available,
            "pubsub": self.pubsub is not None,
            "streams": self.streams is not None,
        }

    async def health_check(self) -> dict:
        """Check health of Redis and all modules.

        Returns:
            Dict with overall health and per-module status.
        """
        result = {
            "healthy": False,
            "connected": self._connected,
            "modules": self._get_module_status(),
            "latency_ms": None,
            "errors": [],
        }

        if not self._connected or not self.redis:
            result["errors"].append("Not connected")
            return result

        try:
            import time
            start = time.time()
            await asyncio.wait_for(self.redis.ping(), timeout=5.0)
            result["latency_ms"] = int((time.time() - start) * 1000)
            result["healthy"] = True
        except Exception as e:
            result["errors"].append(f"Ping failed: {e}")

        # Check individual modules
        if self.timeseries and self.timeseries_available:
            try:
                await self.timeseries.health_check()
            except Exception as e:
                result["errors"].append(f"TimeSeries: {e}")

        return result

    async def close(self) -> None:
        """Close Redis connection and cleanup."""
        if self.pubsub:
            await self.pubsub.close()

        if self.redis:
            await self.redis.aclose()
            self._connected = False
            logger.info("RedisStack: Connection closed")

    # =========================================================================
    # Convenience Methods (delegate to module wrappers)
    # =========================================================================

    async def record_query_metric(
        self,
        latency_ms: float,
        cache_hit: bool = False,
        agent: str | None = None,
        error: bool = False,
    ) -> None:
        """Record query metrics to TimeSeries.

        Falls back to logging if TimeSeries unavailable.
        """
        if self.timeseries and self.timeseries_available:
            await self.timeseries.record_query(latency_ms, cache_hit, agent, error)
        else:
            # Fallback: just log
            logger.debug(f"Query metric: {latency_ms}ms, cache={cache_hit}, agent={agent}")

    async def record_query_event(
        self,
        query: str,
        response: str,
        latency_ms: int,
        agents: list[str],
        cache_hit: bool,
        trace_id: str = "",
        session_id: str = "",
        success: bool = True,
        error: str | None = None,
    ) -> str | None:
        """Record query to audit stream.

        Returns:
            Event ID if recorded, None if streams unavailable.
        """
        if self.streams:
            from .models import QueryEvent
            event = QueryEvent(
                query=query,
                response_preview=response[:500] if response else "",
                latency_ms=latency_ms,
                agent_chain=agents,
                cache_hit=cache_hit,
                trace_id=trace_id,
                session_id=session_id,
                success=success,
                error=error,
            )
            return await self.streams.add_query_event(event)
        return None

    async def publish_index_progress(
        self,
        index_name: str,
        indexed: int,
        total: int,
        errors: int = 0,
        status: str = "running",
        elapsed: float = 0.0,
        rate: float = 0.0,
    ) -> None:
        """Publish indexing progress to Pub/Sub.

        Falls back to logging if Pub/Sub unavailable.
        """
        if self.pubsub:
            from .models import IndexProgress
            progress = IndexProgress(
                index_name=index_name,
                indexed=indexed,
                total=total,
                errors=errors,
                status=status,
                elapsed_seconds=elapsed,
                rate_per_second=rate,
            )
            await self.pubsub.publish_index_progress(progress)
        else:
            logger.info(f"Index progress: {index_name} {indexed}/{total} ({status})")

    async def check_query_seen(self, query: str) -> bool:
        """Check if query was recently seen using Bloom filter.

        Returns False if Bloom unavailable (conservative - don't skip).
        """
        if self.bloom and self.bloom_available:
            return await self.bloom.query_seen(query)
        return False

    async def mark_query_seen(self, query: str) -> None:
        """Mark query as seen in Bloom filter."""
        if self.bloom and self.bloom_available:
            await self.bloom.add_query(query)

    async def track_trending_query(self, query: str) -> None:
        """Track query for trending analysis."""
        if self.bloom and self.bloom_available:
            await self.bloom.track_query_frequency(query)

    async def get_trending_queries(self, k: int = 10) -> list[tuple[str, int]]:
        """Get top K trending queries."""
        if self.bloom and self.bloom_available:
            return await self.bloom.get_top_queries(k)
        return []
