"""ICDA Redis Stack Integration.

Provides high-level abstractions for Redis Stack modules:
- TimeSeries: Metrics recording and time-based analytics
- RediSearch: Full-text and vector similarity search
- ReJSON: Session persistence and structured data
- Bloom: Probabilistic duplicate detection and rate limiting
"""

from .client import RedisStackClient, RedisStackConfig, RedisModule
from .timeseries import TimeSeriesMetrics, MetricType, MetricsRecorder
from .search import QuerySearchIndex, SimilaritySearch, IndexedQuery, QueryIntent
from .json_store import SessionStore, PersistentSession, QueryResultStore
from .bloom import BloomFilters, BloomFilterType

__all__ = [
    # Client
    "RedisStackClient",
    "RedisStackConfig",
    "RedisModule",
    # TimeSeries
    "TimeSeriesMetrics",
    "MetricType",
    "MetricsRecorder",
    # Search
    "QuerySearchIndex",
    "SimilaritySearch",
    "IndexedQuery",
    "QueryIntent",
    # JSON Store
    "SessionStore",
    "PersistentSession",
    "QueryResultStore",
    # Bloom
    "BloomFilters",
    "BloomFilterType",
]
