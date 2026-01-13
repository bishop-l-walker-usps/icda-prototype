"""Redis Stack integration for ICDA.

Provides unified access to all Redis Stack modules:
- RediSearch: Full-text search with facets and suggestions
- RedisJSON: Structured JSON document storage
- RedisTimeSeries: Time-series metrics and analytics
- RedisBloom: Probabilistic data structures (Bloom, Cuckoo, TopK)
- Pub/Sub: Real-time event notifications
- Streams: Event sourcing and audit trails

All modules support graceful degradation when unavailable.
"""

from .client import RedisStackClient
from .models import (
    ModuleStatus,
    MetricSample,
    QueryEvent,
    CustomerEvent,
    IndexProgress,
)
from .router import router, configure_router

__all__ = [
    "RedisStackClient",
    "ModuleStatus",
    "MetricSample",
    "QueryEvent",
    "CustomerEvent",
    "IndexProgress",
    "router",
    "configure_router",
]
