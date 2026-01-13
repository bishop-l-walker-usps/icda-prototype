"""TimeSeries Metrics - Redis TimeSeries for observability.

Tracks:
- Query latency (total and per-agent)
- Cache hit/miss rates
- Token usage and costs
- Error rates by category
- Address verification success rates
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RedisStackClient


class MetricType(str, Enum):
    """Types of metrics tracked by TimeSeries."""
    # Latency metrics (values in milliseconds)
    LATENCY_TOTAL = "latency:total"
    LATENCY_INTENT = "latency:agent:intent"
    LATENCY_CONTEXT = "latency:agent:context"
    LATENCY_PARSER = "latency:agent:parser"
    LATENCY_RESOLVER = "latency:agent:resolver"
    LATENCY_SEARCH = "latency:agent:search"
    LATENCY_KNOWLEDGE = "latency:agent:knowledge"
    LATENCY_NOVA = "latency:agent:nova"
    LATENCY_ENFORCER = "latency:agent:enforcer"
    LATENCY_EMBEDDING = "latency:embedding"

    # Cache metrics (values are counts)
    CACHE_HIT = "cache:hit"
    CACHE_MISS = "cache:miss"

    # Query metrics
    QUERY_VOLUME = "query:volume"
    QUERY_LOOKUP = "query:type:lookup"
    QUERY_COMPLEX = "query:type:complex"
    QUERY_BLOCKED = "query:type:blocked"

    # Error metrics
    ERROR_TOTAL = "error:total"
    ERROR_TIMEOUT = "error:timeout"
    ERROR_VALIDATION = "error:validation"
    ERROR_BEDROCK = "error:bedrock"

    # Token/cost metrics
    TOKENS_INPUT = "tokens:input"
    TOKENS_OUTPUT = "tokens:output"

    # Address verification
    ADDRESS_SUCCESS = "address:success"
    ADDRESS_PARTIAL = "address:partial"
    ADDRESS_FAILED = "address:failed"


@dataclass(slots=True)
class MetricDataPoint:
    """A single metric data point."""
    metric: MetricType
    value: float
    timestamp_ms: int | None = None
    labels: dict[str, str] | None = None


@dataclass(slots=True)
class AggregatedMetric:
    """Aggregated metric result."""
    metric: MetricType
    aggregation: str  # avg, sum, min, max, count
    value: float
    start_ts: int
    end_ts: int
    bucket_size_ms: int
    data_points: list[tuple[int, float]]  # (timestamp, value) pairs


class TimeSeriesMetrics:
    """Redis TimeSeries metrics collector and aggregator.

    Provides:
    - Metric recording with automatic timestamp
    - Labeled metrics for filtering
    - Time-range queries with aggregation
    - SLA compliance calculations
    """

    __slots__ = ("_client", "_prefix", "_retention_ms", "_enabled")

    def __init__(
        self,
        client: RedisStackClient,
        prefix: str = "icda:metrics",
        retention_ms: int | None = None,
    ):
        """Initialize TimeSeries metrics.

        Args:
            client: Redis Stack client.
            prefix: Key prefix for all metrics.
            retention_ms: Data retention in ms. Uses config default if None.
        """
        self._client = client
        self._prefix = prefix
        self._retention_ms = retention_ms or client.config.metrics_retention_ms

    @property
    def enabled(self) -> bool:
        """Check if TimeSeries is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.TIMESERIES)

    def _key(self, metric: MetricType) -> str:
        """Generate key for a metric."""
        return f"{self._prefix}:{metric.value}"

    async def ensure_series(self, metric: MetricType, labels: dict[str, str] | None = None) -> bool:
        """Ensure a time series exists with proper configuration.

        Args:
            metric: Metric type to create.
            labels: Optional labels for filtering.

        Returns:
            True if series exists or was created.
        """
        if not self.enabled:
            return False

        key = self._key(metric)
        try:
            # Check if series exists
            if await self._client.exists(key):
                return True

            # Create with labels
            cmd = [
                "TS.CREATE", key,
                "RETENTION", str(self._retention_ms),
                "DUPLICATE_POLICY", "LAST",
            ]

            if labels:
                cmd.append("LABELS")
                for k, v in labels.items():
                    cmd.extend([k, v])
            else:
                # Default labels
                cmd.extend(["LABELS", "app", "icda", "type", metric.value.split(":")[0]])

            await self._client.execute(*cmd)
            return True
        except Exception as e:
            # Series might already exist with different config
            if "already exists" in str(e).lower():
                return True
            print(f"TimeSeries: Failed to create {key}: {e}")
            return False

    async def record(
        self,
        metric: MetricType,
        value: float,
        timestamp_ms: int | None = None,
    ) -> bool:
        """Record a metric value.

        Args:
            metric: Metric type.
            value: Metric value.
            timestamp_ms: Optional timestamp in ms. Uses current time if None.

        Returns:
            True if recorded successfully.
        """
        if not self.enabled:
            return False

        key = self._key(metric)
        ts = timestamp_ms or int(time() * 1000)

        try:
            # Ensure series exists
            await self.ensure_series(metric)

            # Add data point
            await self._client.execute("TS.ADD", key, ts, value)
            return True
        except Exception as e:
            print(f"TimeSeries: Failed to record {metric.value}: {e}")
            return False

    async def record_batch(self, data_points: list[MetricDataPoint]) -> int:
        """Record multiple metrics in a batch.

        Args:
            data_points: List of metric data points.

        Returns:
            Number of successfully recorded points.
        """
        if not self.enabled or not data_points:
            return 0

        success = 0
        ts = int(time() * 1000)

        try:
            pipe = await self._client.pipeline()

            for dp in data_points:
                key = self._key(dp.metric)
                point_ts = dp.timestamp_ms or ts
                pipe.execute_command("TS.ADD", key, point_ts, dp.value, "ON_DUPLICATE", "LAST")

            results = await pipe.execute()
            success = sum(1 for r in results if r is not None)
        except Exception as e:
            print(f"TimeSeries: Batch record failed: {e}")

        return success

    async def increment(self, metric: MetricType, amount: float = 1.0) -> bool:
        """Increment a counter metric.

        Args:
            metric: Metric type.
            amount: Amount to increment by.

        Returns:
            True if incremented successfully.
        """
        if not self.enabled:
            return False

        key = self._key(metric)

        try:
            await self.ensure_series(metric)
            await self._client.execute("TS.INCRBY", key, amount)
            return True
        except Exception as e:
            print(f"TimeSeries: Failed to increment {metric.value}: {e}")
            return False

    async def get_latest(self, metric: MetricType) -> tuple[int, float] | None:
        """Get the latest value for a metric.

        Args:
            metric: Metric type.

        Returns:
            Tuple of (timestamp_ms, value) or None.
        """
        if not self.enabled:
            return None

        try:
            result = await self._client.execute("TS.GET", self._key(metric))
            if result:
                return (int(result[0]), float(result[1]))
        except Exception:
            pass
        return None

    async def get_range(
        self,
        metric: MetricType,
        start_ms: int | str = "-",
        end_ms: int | str = "+",
        aggregation: str | None = None,
        bucket_size_ms: int = 60000,
        count: int | None = None,
    ) -> list[tuple[int, float]]:
        """Get metric values over a time range.

        Args:
            metric: Metric type.
            start_ms: Start timestamp in ms, or "-" for earliest.
            end_ms: End timestamp in ms, or "+" for latest.
            aggregation: Aggregation type (avg, sum, min, max, count).
            bucket_size_ms: Bucket size for aggregation.
            count: Maximum number of results.

        Returns:
            List of (timestamp_ms, value) tuples.
        """
        if not self.enabled:
            return []

        try:
            cmd = ["TS.RANGE", self._key(metric), str(start_ms), str(end_ms)]

            if aggregation:
                cmd.extend(["AGGREGATION", aggregation, str(bucket_size_ms)])

            if count:
                cmd.extend(["COUNT", str(count)])

            result = await self._client.execute(*cmd)
            return [(int(r[0]), float(r[1])) for r in result] if result else []
        except Exception as e:
            print(f"TimeSeries: Range query failed: {e}")
            return []

    async def get_percentile(
        self,
        metric: MetricType,
        percentile: float,
        start_ms: int | str = "-",
        end_ms: int | str = "+",
    ) -> float | None:
        """Calculate percentile for a metric over a time range.

        Args:
            metric: Metric type.
            percentile: Percentile (0-100).
            start_ms: Start timestamp.
            end_ms: End timestamp.

        Returns:
            Percentile value or None.
        """
        if not self.enabled:
            return None

        # Get all data points and calculate percentile
        data = await self.get_range(metric, start_ms, end_ms)
        if not data:
            return None

        values = sorted(v for _, v in data)
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]

    async def get_cache_hit_rate(
        self,
        start_ms: int | str = "-",
        end_ms: int | str = "+",
    ) -> float | None:
        """Calculate cache hit rate over a time range.

        Args:
            start_ms: Start timestamp.
            end_ms: End timestamp.

        Returns:
            Hit rate (0.0-1.0) or None.
        """
        if not self.enabled:
            return None

        hits = await self.get_range(MetricType.CACHE_HIT, start_ms, end_ms, "sum", 86400000)
        misses = await self.get_range(MetricType.CACHE_MISS, start_ms, end_ms, "sum", 86400000)

        total_hits = sum(v for _, v in hits)
        total_misses = sum(v for _, v in misses)
        total = total_hits + total_misses

        return total_hits / total if total > 0 else None

    async def get_sla_compliance(
        self,
        threshold_ms: float = 3000,
        start_ms: int | str = "-",
        end_ms: int | str = "+",
    ) -> float | None:
        """Calculate SLA compliance (% queries under threshold).

        Args:
            threshold_ms: Latency threshold in ms.
            start_ms: Start timestamp.
            end_ms: End timestamp.

        Returns:
            Compliance rate (0.0-1.0) or None.
        """
        if not self.enabled:
            return None

        data = await self.get_range(MetricType.LATENCY_TOTAL, start_ms, end_ms)
        if not data:
            return None

        under_threshold = sum(1 for _, v in data if v < threshold_ms)
        return under_threshold / len(data)

    async def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get all metrics for dashboard display.

        Returns:
            Dict with dashboard metrics.
        """
        now_ms = int(time() * 1000)
        hour_ago = now_ms - 3600000
        day_ago = now_ms - 86400000

        return {
            "cache": {
                "hit_rate_1h": await self.get_cache_hit_rate(hour_ago, now_ms),
                "hit_rate_24h": await self.get_cache_hit_rate(day_ago, now_ms),
            },
            "latency": {
                "p50_1h": await self.get_percentile(MetricType.LATENCY_TOTAL, 50, hour_ago, now_ms),
                "p95_1h": await self.get_percentile(MetricType.LATENCY_TOTAL, 95, hour_ago, now_ms),
                "p99_1h": await self.get_percentile(MetricType.LATENCY_TOTAL, 99, hour_ago, now_ms),
            },
            "sla": {
                "compliance_1h": await self.get_sla_compliance(3000, hour_ago, now_ms),
                "compliance_24h": await self.get_sla_compliance(3000, day_ago, now_ms),
            },
            "volume": {
                "queries_1h": len(await self.get_range(MetricType.QUERY_VOLUME, hour_ago, now_ms)),
            },
        }


class MetricsRecorder:
    """Context manager for recording pipeline metrics.

    Usage:
        async with MetricsRecorder(metrics, "intent") as recorder:
            # ... do work ...
            recorder.set_success(True)
    """

    __slots__ = ("_metrics", "_agent", "_start_time", "_success", "_tokens_in", "_tokens_out")

    def __init__(self, metrics: TimeSeriesMetrics, agent: str):
        self._metrics = metrics
        self._agent = agent
        self._start_time: float = 0
        self._success = True
        self._tokens_in = 0
        self._tokens_out = 0

    async def __aenter__(self) -> MetricsRecorder:
        self._start_time = time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        latency_ms = (time() - self._start_time) * 1000

        # Record agent latency
        metric_map = {
            "intent": MetricType.LATENCY_INTENT,
            "context": MetricType.LATENCY_CONTEXT,
            "parser": MetricType.LATENCY_PARSER,
            "resolver": MetricType.LATENCY_RESOLVER,
            "search": MetricType.LATENCY_SEARCH,
            "knowledge": MetricType.LATENCY_KNOWLEDGE,
            "nova": MetricType.LATENCY_NOVA,
            "enforcer": MetricType.LATENCY_ENFORCER,
            "embedding": MetricType.LATENCY_EMBEDDING,
            "total": MetricType.LATENCY_TOTAL,
        }

        if metric := metric_map.get(self._agent):
            await self._metrics.record(metric, latency_ms)

        # Record tokens if set
        if self._tokens_in > 0:
            await self._metrics.record(MetricType.TOKENS_INPUT, self._tokens_in)
        if self._tokens_out > 0:
            await self._metrics.record(MetricType.TOKENS_OUTPUT, self._tokens_out)

        # Record errors if exception occurred
        if exc_type is not None:
            await self._metrics.increment(MetricType.ERROR_TOTAL)

    def set_success(self, success: bool) -> None:
        """Set success status."""
        self._success = success

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts for cost tracking."""
        self._tokens_in = input_tokens
        self._tokens_out = output_tokens
