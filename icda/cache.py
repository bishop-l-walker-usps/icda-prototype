"""Caching with Redis or in-memory fallback.

Works in LITE MODE (no Redis) with automatic in-memory fallback.
Supports health-aware caching - bypasses cache when index is unhealthy.
Automatically reconnects to Redis if connection is lost.
"""

from functools import cache
from hashlib import sha256
from time import time
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """Async cache with Redis or in-memory fallback.

    Supports health-aware caching that bypasses cache when the
    search index is unhealthy to prevent serving stale results.
    Automatically attempts to reconnect to Redis on failure.
    """

    __slots__ = (
        "client", "ttl", "available", "_fallback",
        "_index_healthy", "_last_health_update",
        "_redis_url", "_last_reconnect_attempt", "_reconnect_interval"
    )

    def __init__(self, ttl: int = 43200, reconnect_interval: int = 30):
        self.ttl = ttl
        self.client = None
        self.available = False
        self._fallback: dict[str, tuple[str, float]] = {}
        self._index_healthy = True  # Assume healthy until told otherwise
        self._last_health_update = 0.0
        self._redis_url = ""
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval = reconnect_interval  # seconds between reconnection attempts

    async def connect(self, url: str) -> None:
        """Connect to Redis or use in-memory fallback."""
        self._redis_url = url
        if not url:
            logger.info("Cache: Using in-memory (no Redis configured)")
            return

        try:
            import redis.asyncio as aioredis
            self.client = aioredis.from_url(
                url,
                decode_responses=True,
                socket_timeout=5.0,           # 5s timeout for read/write operations
                socket_connect_timeout=5.0,   # 5s timeout for connection
                health_check_interval=30,     # Periodic health checks
            )
            await self.client.ping()
            self.available = True
            logger.info(f"Cache: Redis connected ({url})")
        except ImportError:
            logger.warning("Cache: redis package not installed, using in-memory")
        except Exception as e:
            logger.warning(f"Cache: Redis unavailable ({e}), using in-memory")

    async def reconnect(self) -> bool:
        """Attempt to reconnect to Redis if not connected.

        Returns:
            True if reconnected successfully, False otherwise.
        """
        if self.available:
            return True  # Already connected

        if not self._redis_url:
            return False  # No Redis URL configured

        # Rate limit reconnection attempts
        current_time = time()
        if current_time - self._last_reconnect_attempt < self._reconnect_interval:
            return False

        self._last_reconnect_attempt = current_time

        try:
            import redis.asyncio as aioredis
            self.client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                health_check_interval=30,
            )
            await self.client.ping()
            self.available = True
            # Migrate in-memory cache to Redis
            migrated = 0
            for key, (value, expires) in list(self._fallback.items()):
                if time() < expires:
                    remaining_ttl = int(expires - time())
                    if remaining_ttl > 0:
                        try:
                            await self.client.setex(key, remaining_ttl, value)
                            migrated += 1
                        except Exception:
                            pass
            self._fallback.clear()
            logger.info(f"Cache: Redis reconnected, migrated {migrated} cached entries")
            return True
        except Exception as e:
            logger.debug(f"Cache: Redis reconnection failed - {e}")
            return False

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

    async def get(self, key: str) -> str | None:
        if self.available:
            try:
                return await self.client.get(key)
            except Exception:
                pass
        # In-memory fallback
        if (entry := self._fallback.get(key)) and time() < entry[1]:
            return entry[0]
        self._fallback.pop(key, None)
        return None

    async def set(self, key: str, value: str) -> None:
        if self.available:
            try:
                await self.client.setex(key, self.ttl, value)
                return
            except Exception:
                pass
        # In-memory fallback
        self._fallback[key] = (value, time() + self.ttl)

    async def clear(self) -> None:
        if self.available:
            try:
                await self.client.flushdb()
                return
            except Exception:
                pass
        self._fallback.clear()

    async def stats(self) -> dict:
        if self.available:
            try:
                info = await self.client.info("keyspace")
                return {
                    "keys": info.get("db0", {}).get("keys", 0),
                    "backend": "redis",
                    "ttl_hours": self.ttl // 3600
                }
            except Exception:
                self.available = False
        return {
            "keys": len(self._fallback),
            "backend": "memory",
            "ttl_hours": self.ttl // 3600
        }

    @staticmethod
    @cache
    def make_key(query: str) -> str:
        return f"icda:q:{sha256(query.casefold().strip().encode()).hexdigest()[:16]}"

    # =========================================================================
    # Health-Aware Caching Methods
    # =========================================================================

    def set_index_health(self, healthy: bool) -> None:
        """Update index health status from external health check.

        Args:
            healthy: Whether the search index is healthy.
        """
        self._index_healthy = healthy
        self._last_health_update = time()

    @property
    def index_healthy(self) -> bool:
        """Check if the search index is healthy."""
        return self._index_healthy

    @property
    def should_use_cache(self) -> bool:
        """Check if cache should be used based on index health.

        Returns False if index is unhealthy, preventing stale results.
        """
        return self._index_healthy

    async def get_if_healthy(self, key: str) -> str | None:
        """Get from cache only if index is healthy.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if unhealthy/not found.
        """
        if not self._index_healthy:
            return None  # Bypass cache when index unhealthy
        return await self.get(key)

    async def set_if_healthy(self, key: str, value: str) -> None:
        """Set in cache only if index is healthy.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if not self._index_healthy:
            return  # Don't cache results from fallback mode
        await self.set(key, value)
