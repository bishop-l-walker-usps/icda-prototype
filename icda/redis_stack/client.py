"""Redis Stack Client - Connection management and module detection.

Provides:
- Unified connection to Redis Stack
- Module availability detection (TimeSeries, Search, JSON, Bloom)
- Health checks and memory monitoring
- Graceful degradation when modules unavailable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import redis.asyncio as redis


class RedisModule(str, Enum):
    """Available Redis Stack modules."""
    TIMESERIES = "timeseries"
    SEARCH = "search"
    JSON = "ReJSON"
    BLOOM = "bf"


@dataclass
class RedisStackConfig:
    """Configuration for Redis Stack connection."""
    url: str = "redis://localhost:6379"
    max_connections: int = 20
    metrics_retention_days: int = 30
    session_ttl_days: int = 7
    embedding_dimensions: int = 1024

    @property
    def metrics_retention_ms(self) -> int:
        """Convert retention days to milliseconds."""
        return self.metrics_retention_days * 24 * 60 * 60 * 1000

    @property
    def session_ttl_seconds(self) -> int:
        """Convert session TTL to seconds."""
        return self.session_ttl_days * 24 * 60 * 60


class RedisStackClient:
    """Redis Stack client with module detection and health monitoring.

    Usage:
        config = RedisStackConfig(url="redis://localhost:6379")
        client = RedisStackClient(config)
        await client.connect()

        # Check module availability
        if client.has_module(RedisModule.TIMESERIES):
            # Use TimeSeries features
            pass

        # Health check
        health = await client.health_check()
    """

    __slots__ = ("_config", "_client", "_available", "_modules")

    def __init__(self, config: RedisStackConfig | None = None):
        """Initialize Redis Stack client.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self._config = config or RedisStackConfig()
        self._client: redis.Redis | None = None
        self._available = False
        self._modules: dict[RedisModule, bool] = {
            module: False for module in RedisModule
        }

    @property
    def config(self) -> RedisStackConfig:
        """Get configuration."""
        return self._config

    @property
    def client(self) -> redis.Redis | None:
        """Get underlying Redis client."""
        return self._client

    @property
    def available(self) -> bool:
        """Check if client is connected and available."""
        return self._available

    def has_module(self, module: RedisModule) -> bool:
        """Check if a specific module is available.

        Args:
            module: The Redis module to check.

        Returns:
            True if module is available.
        """
        return self._modules.get(module, False)

    async def connect(self) -> bool:
        """Connect to Redis and detect available modules.

        Returns:
            True if connection successful.
        """
        try:
            self._client = redis.from_url(
                self._config.url,
                max_connections=self._config.max_connections,
                decode_responses=True,
            )

            # Test connection
            await self._client.ping()

            # Detect modules
            await self._detect_modules()

            self._available = True
            print(f"RedisStack: Connected to {self._config.url}")
            return True
        except Exception as e:
            print(f"RedisStack: Connection failed - {e}")
            self._available = False
            return False

    async def _detect_modules(self) -> None:
        """Detect available Redis Stack modules."""
        if not self._client:
            return

        try:
            module_list = await self._client.module_list()
            module_names = {m.get("name", "").lower() for m in module_list}

            # Map module names to enum
            self._modules[RedisModule.TIMESERIES] = "timeseries" in module_names
            self._modules[RedisModule.SEARCH] = "search" in module_names
            self._modules[RedisModule.JSON] = "rejson" in module_names
            self._modules[RedisModule.BLOOM] = "bf" in module_names

            available_names = [m.value for m in RedisModule if self._modules[m]]
            if available_names:
                print(f"RedisStack: Modules available: {', '.join(available_names)}")
            else:
                print("RedisStack: No Stack modules detected (basic Redis mode)")
        except Exception as e:
            print(f"RedisStack: Module detection failed - {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on Redis Stack.

        Returns:
            Dict with health status and metrics.
        """
        if not self._available or not self._client:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Not connected",
            }

        try:
            info = await self._client.info()

            return {
                "status": "healthy",
                "connected": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "memory_peak": info.get("used_memory_peak_human", "unknown"),
                "modules": {
                    module.value: self._modules[module]
                    for module in RedisModule
                },
                "clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": True,
                "error": str(e),
            }

    async def execute(self, *args) -> Any:
        """Execute a raw Redis command.

        Args:
            *args: Command and arguments.

        Returns:
            Command result.
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return await self._client.execute_command(*args)

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: Key to check.

        Returns:
            True if key exists.
        """
        if not self._client:
            return False
        return await self._client.exists(key) > 0

    async def pipeline(self) -> redis.client.Pipeline:
        """Get a pipeline for batched commands.

        Returns:
            Redis pipeline.
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.pipeline()

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._available = False
            print("RedisStack: Connection closed")
