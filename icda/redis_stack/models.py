"""Shared models for Redis Stack integration."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModuleStatus(str, Enum):
    """Redis module availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class MetricSample:
    """A single time-series metric sample."""
    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class QueryEvent:
    """Query audit event for Streams."""
    query: str
    response_preview: str
    latency_ms: int
    agent_chain: list[str]
    cache_hit: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""
    session_id: str = ""
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response_preview": self.response_preview[:200] if self.response_preview else "",
            "latency_ms": str(self.latency_ms),
            "agent_chain": ",".join(self.agent_chain),
            "cache_hit": "1" if self.cache_hit else "0",
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "success": "1" if self.success else "0",
            "error": self.error or "",
        }


@dataclass
class CustomerEvent:
    """Customer change event for Streams."""
    crid: str
    action: str  # created, updated, deleted
    changes: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    actor: str = "system"

    def to_dict(self) -> dict[str, Any]:
        import json
        return {
            "crid": self.crid,
            "action": self.action,
            "changes": json.dumps(self.changes),
            "timestamp": self.timestamp,
            "actor": self.actor,
        }


@dataclass
class IndexProgress:
    """Indexing progress event for Pub/Sub."""
    index_name: str
    indexed: int
    total: int
    errors: int = 0
    status: str = "running"  # running, completed, failed
    elapsed_seconds: float = 0.0
    rate_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "indexed": self.indexed,
            "total": self.total,
            "errors": self.errors,
            "percent": round((self.indexed / self.total) * 100, 1) if self.total > 0 else 0,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "rate_per_second": round(self.rate_per_second, 2),
        }


@dataclass
class SearchSuggestion:
    """Autocomplete suggestion result."""
    text: str
    score: float
    payload: str | None = None


@dataclass
class FacetResult:
    """Faceted search result."""
    field: str
    values: list[tuple[str, int]]  # (value, count)


@dataclass
class TrendingQuery:
    """Trending query from TopK."""
    query: str
    count: int
    rank: int
