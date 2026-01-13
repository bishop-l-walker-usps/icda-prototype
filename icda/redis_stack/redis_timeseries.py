"""RedisTimeSeries wrapper for metrics and analytics.

Tracks:
- Query volume (requests/min)
- Response latency (p50, p95, p99)
- Cache hit rate
- Per-agent performance
- Error rates
- Indexing throughput
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RedisTimeSeriesWrapper:
    """Wrapper for RedisTimeSeries operations.

    Key naming convention:
        ts:queries:volume          - Query count per minute
        ts:queries:latency         - Query latency (ms)
        ts:cache:hits              - Cache hit count
        ts:cache:misses            - Cache miss count
        ts:agent:{name}:latency    - Per-agent latency
        ts:errors:count            - Error count
        ts:index:throughput        - Docs indexed per second
    """

    # Retention policies (ms)
    RETENTION_1H = 3600 * 1000        # 1 hour detailed
    RETENTION_24H = 86400 * 1000      # 24 hours
    RETENTION_7D = 604800 * 1000      # 7 days aggregated

    # Aggregation rules
    AGGREGATIONS = [
        ("1m", 60000, "avg"),      # 1 minute buckets
        ("5m", 300000, "avg"),     # 5 minute buckets
        ("1h", 3600000, "avg"),    # 1 hour buckets
    ]

    def __init__(self, redis):
        self.redis = redis

    async def ensure_timeseries(self) -> None:
        """Create time series keys with proper configuration."""
        timeseries_config = [
            ("ts:queries:volume", self.RETENTION_24H, {"type": "queries", "metric": "volume"}),
            ("ts:queries:latency", self.RETENTION_24H, {"type": "queries", "metric": "latency"}),
            ("ts:cache:hits", self.RETENTION_24H, {"type": "cache", "metric": "hits"}),
            ("ts:cache:misses", self.RETENTION_24H, {"type": "cache", "metric": "misses"}),
            ("ts:errors:count", self.RETENTION_7D, {"type": "errors", "metric": "count"}),
            ("ts:index:throughput", self.RETENTION_24H, {"type": "index", "metric": "throughput"}),
        ]

        for key, retention, labels in timeseries_config:
            await self._ensure_ts(key, retention, labels)

    async def _ensure_ts(
        self,
        key: str,
        retention: int,
        labels: dict[str, str],
        duplicate_policy: str = "last"
    ) -> bool:
        """Create time series if it doesn't exist."""
        try:
            # Check if exists
            await self.redis.execute_command("TS.INFO", key)
            return True
        except Exception:
            pass

        try:
            # Create with labels
            label_args = []
            for k, v in labels.items():
                label_args.extend(["LABELS", k, v])

            await self.redis.execute_command(
                "TS.CREATE", key,
                "RETENTION", retention,
                "DUPLICATE_POLICY", duplicate_policy,
                *label_args
            )
            logger.debug(f"Created time series: {key}")
            return True
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Failed to create time series {key}: {e}")
            return False

    async def record_query(
        self,
        latency_ms: float,
        cache_hit: bool = False,
        agent: str | None = None,
        error: bool = False,
    ) -> None:
        """Record a query execution.

        Args:
            latency_ms: Query latency in milliseconds
            cache_hit: Whether result was from cache
            agent: Name of agent that processed query
            error: Whether query resulted in error
        """
        ts = int(time.time() * 1000)

        try:
            # Record volume (increment by 1)
            await self.redis.execute_command(
                "TS.ADD", "ts:queries:volume", ts, 1
            )

            # Record latency
            await self.redis.execute_command(
                "TS.ADD", "ts:queries:latency", ts, latency_ms
            )

            # Record cache hit/miss
            if cache_hit:
                await self.redis.execute_command(
                    "TS.ADD", "ts:cache:hits", ts, 1
                )
            else:
                await self.redis.execute_command(
                    "TS.ADD", "ts:cache:misses", ts, 1
                )

            # Record error
            if error:
                await self.redis.execute_command(
                    "TS.ADD", "ts:errors:count", ts, 1
                )

            # Record per-agent latency
            if agent:
                agent_key = f"ts:agent:{agent}:latency"
                await self._ensure_ts(
                    agent_key,
                    self.RETENTION_24H,
                    {"type": "agent", "agent": agent, "metric": "latency"}
                )
                await self.redis.execute_command(
                    "TS.ADD", agent_key, ts, latency_ms
                )

        except Exception as e:
            logger.debug(f"Failed to record query metric: {e}")

    async def record_index_throughput(self, docs_indexed: int) -> None:
        """Record indexing throughput."""
        try:
            ts = int(time.time() * 1000)
            await self.redis.execute_command(
                "TS.ADD", "ts:index:throughput", ts, docs_indexed
            )
        except Exception as e:
            logger.debug(f"Failed to record index throughput: {e}")

    async def get_query_stats(self, range_ms: int = 3600000) -> dict[str, Any]:
        """Get query statistics for time range.

        Args:
            range_ms: Time range in milliseconds (default 1 hour)

        Returns:
            Dict with volume, latency percentiles, cache rate
        """
        now = int(time.time() * 1000)
        start = now - range_ms

        result = {
            "range_ms": range_ms,
            "query_count": 0,
            "latency_avg": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "cache_hit_rate": 0,
            "error_count": 0,
        }

        try:
            # Get query volume
            volume = await self.redis.execute_command(
                "TS.RANGE", "ts:queries:volume", start, now
            )
            result["query_count"] = sum(float(v[1]) for v in volume) if volume else 0

            # Get latency stats
            latency = await self.redis.execute_command(
                "TS.RANGE", "ts:queries:latency", start, now
            )
            if latency:
                values = sorted([float(v[1]) for v in latency])
                n = len(values)
                result["latency_avg"] = sum(values) / n
                result["latency_p50"] = values[int(n * 0.5)] if n > 0 else 0
                result["latency_p95"] = values[int(n * 0.95)] if n > 0 else 0
                result["latency_p99"] = values[int(n * 0.99)] if n > 0 else 0

            # Get cache hit rate
            hits = await self.redis.execute_command(
                "TS.RANGE", "ts:cache:hits", start, now
            )
            misses = await self.redis.execute_command(
                "TS.RANGE", "ts:cache:misses", start, now
            )
            total_hits = sum(float(v[1]) for v in hits) if hits else 0
            total_misses = sum(float(v[1]) for v in misses) if misses else 0
            total = total_hits + total_misses
            result["cache_hit_rate"] = (total_hits / total * 100) if total > 0 else 0

            # Get error count
            errors = await self.redis.execute_command(
                "TS.RANGE", "ts:errors:count", start, now
            )
            result["error_count"] = sum(float(v[1]) for v in errors) if errors else 0

        except Exception as e:
            logger.warning(f"Failed to get query stats: {e}")

        return result

    async def get_agent_stats(self, range_ms: int = 3600000) -> dict[str, dict]:
        """Get per-agent latency statistics.

        Returns:
            Dict mapping agent name to stats
        """
        now = int(time.time() * 1000)
        start = now - range_ms
        result = {}

        try:
            # Find all agent time series
            keys = await self.redis.execute_command(
                "TS.QUERYINDEX", "type=agent"
            )

            for key in keys or []:
                agent_name = key.split(":")[2] if ":" in key else key

                try:
                    data = await self.redis.execute_command(
                        "TS.RANGE", key, start, now
                    )
                    if data:
                        values = [float(v[1]) for v in data]
                        result[agent_name] = {
                            "count": len(values),
                            "avg_latency": sum(values) / len(values),
                            "min_latency": min(values),
                            "max_latency": max(values),
                        }
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Failed to get agent stats: {e}")

        return result

    async def get_time_series(
        self,
        key: str,
        range_ms: int = 3600000,
        bucket_ms: int = 60000,
        aggregation: str = "avg"
    ) -> list[tuple[int, float]]:
        """Get time series data with aggregation.

        Args:
            key: Time series key
            range_ms: Time range in ms
            bucket_ms: Aggregation bucket size in ms
            aggregation: Aggregation type (avg, sum, min, max, count)

        Returns:
            List of (timestamp, value) tuples
        """
        now = int(time.time() * 1000)
        start = now - range_ms

        try:
            data = await self.redis.execute_command(
                "TS.RANGE", key, start, now,
                "AGGREGATION", aggregation, bucket_ms
            )
            return [(int(d[0]), float(d[1])) for d in data] if data else []
        except Exception as e:
            logger.warning(f"Failed to get time series {key}: {e}")
            return []

    async def health_check(self) -> dict:
        """Check TimeSeries health."""
        try:
            info = await self.redis.execute_command("TS.INFO", "ts:queries:volume")
            return {"healthy": True, "samples": info}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
