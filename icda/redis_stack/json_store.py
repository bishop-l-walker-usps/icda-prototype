"""ReJSON Session Store - Persistent session storage.

Provides:
- Session persistence across server restarts
- Structured conversation history
- Session analytics and context
- Query result caching with structured data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RedisStackClient


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    msg_id: str
    query: str
    response_summary: str
    customer_count: int
    customers: list[str]
    intent: str
    latency_ms: float
    cache_hit: bool
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "msg_id": self.msg_id,
            "query": self.query,
            "response_summary": self.response_summary,
            "customer_count": self.customer_count,
            "customers": self.customers,
            "intent": self.intent,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMessage:
        """Create from dictionary."""
        return cls(
            msg_id=data.get("msg_id", ""),
            query=data.get("query", ""),
            response_summary=data.get("response_summary", ""),
            customer_count=data.get("customer_count", 0),
            customers=data.get("customers", []),
            intent=data.get("intent", ""),
            latency_ms=data.get("latency_ms", 0.0),
            cache_hit=data.get("cache_hit", False),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class SessionAnalytics:
    """Analytics for a session."""
    total_queries: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    intent_counts: dict[str, int] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "total_latency_ms": self.total_latency_ms,
            "intent_counts": self.intent_counts,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionAnalytics:
        """Create from dictionary."""
        return cls(
            total_queries=data.get("total_queries", 0),
            cache_hits=data.get("cache_hits", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            intent_counts=data.get("intent_counts", {}),
        )


@dataclass
class PersistentSession:
    """A persistent session with conversation history."""
    session_id: str
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)
    conversation: list[ConversationMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    analytics: SessionAnalytics = field(default_factory=SessionAnalytics)

    def add_message(
        self,
        msg_id: str,
        query: str,
        response_summary: str,
        customer_count: int,
        customers: list[str],
        intent: str,
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        """Add a message to the conversation.

        Args:
            msg_id: Message ID.
            query: User query.
            response_summary: Response summary.
            customer_count: Number of customers in response.
            customers: Customer IDs.
            intent: Query intent.
            latency_ms: Query latency.
            cache_hit: Whether cache was hit.
        """
        message = ConversationMessage(
            msg_id=msg_id,
            query=query,
            response_summary=response_summary,
            customer_count=customer_count,
            customers=customers,
            intent=intent,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
        self.conversation.append(message)
        self.last_active = time()

        # Update analytics
        self.analytics.total_queries += 1
        if cache_hit:
            self.analytics.cache_hits += 1
        self.analytics.total_latency_ms += latency_ms
        self.analytics.intent_counts[intent] = self.analytics.intent_counts.get(intent, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "conversation": [m.to_dict() for m in self.conversation],
            "context": self.context,
            "analytics": self.analytics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistentSession:
        """Create from dictionary."""
        session = cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", 0.0),
            last_active=data.get("last_active", 0.0),
            context=data.get("context", {}),
        )
        session.conversation = [
            ConversationMessage.from_dict(m)
            for m in data.get("conversation", [])
        ]
        session.analytics = SessionAnalytics.from_dict(data.get("analytics", {}))
        return session


class SessionStore:
    """ReJSON-based session persistence.

    Stores sessions as JSON documents for:
    - Cross-restart persistence
    - Complex nested data structures
    - Atomic updates
    """

    __slots__ = ("_client", "_prefix", "_ttl_seconds", "_enabled")

    KEY_PREFIX = "icda:session:"

    def __init__(
        self,
        client: RedisStackClient,
        ttl_seconds: int | None = None,
    ):
        """Initialize session store.

        Args:
            client: Redis Stack client.
            ttl_seconds: Session TTL. Uses config default if None.
        """
        self._client = client
        self._prefix = self.KEY_PREFIX
        self._ttl_seconds = ttl_seconds or client.config.session_ttl_seconds

    @property
    def enabled(self) -> bool:
        """Check if ReJSON is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.JSON)

    def _key(self, session_id: str) -> str:
        """Generate key for a session."""
        return f"{self._prefix}{session_id}"

    async def save(self, session: PersistentSession) -> bool:
        """Save a session.

        Args:
            session: Session to save.

        Returns:
            True if saved successfully.
        """
        if not self.enabled:
            return False

        key = self._key(session.session_id)
        try:
            data = json.dumps(session.to_dict())
            await self._client.execute("JSON.SET", key, "$", data)
            await self._client.execute("EXPIRE", key, self._ttl_seconds)
            return True
        except Exception as e:
            print(f"SessionStore: Failed to save session: {e}")
            return False

    async def get(self, session_id: str) -> PersistentSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID.

        Returns:
            Session or None if not found.
        """
        if not self.enabled:
            return None

        key = self._key(session_id)
        try:
            result = await self._client.execute("JSON.GET", key)
            if result:
                data = json.loads(result)
                return PersistentSession.from_dict(data)
        except Exception as e:
            print(f"SessionStore: Failed to get session: {e}")
        return None

    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID.

        Returns:
            True if deleted.
        """
        if not self.enabled:
            return False

        key = self._key(session_id)
        try:
            await self._client.execute("DEL", key)
            return True
        except Exception as e:
            print(f"SessionStore: Failed to delete session: {e}")
            return False

    async def update_context(self, session_id: str, context: dict[str, Any]) -> bool:
        """Update session context.

        Args:
            session_id: Session ID.
            context: Context to merge.

        Returns:
            True if updated.
        """
        if not self.enabled:
            return False

        key = self._key(session_id)
        try:
            for k, v in context.items():
                await self._client.execute(
                    "JSON.SET", key, f"$.context.{k}", json.dumps(v)
                )
            return True
        except Exception as e:
            print(f"SessionStore: Failed to update context: {e}")
            return False

    async def list_sessions(self, limit: int = 100) -> list[str]:
        """List all session IDs.

        Args:
            limit: Maximum sessions to return.

        Returns:
            List of session IDs.
        """
        if not self.enabled:
            return []

        try:
            keys = []
            cursor = "0"
            while True:
                result = await self._client.execute(
                    "SCAN", cursor, "MATCH", f"{self._prefix}*", "COUNT", "100"
                )
                cursor = result[0]
                keys.extend(result[1])
                if cursor == "0" or len(keys) >= limit:
                    break
            return [k.replace(self._prefix, "") for k in keys[:limit]]
        except Exception as e:
            print(f"SessionStore: Failed to list sessions: {e}")
            return []


class QueryResultStore:
    """ReJSON-based query result cache.

    Stores complex query results as JSON for:
    - Enhanced caching with structured data
    - Result pagination
    - Incremental updates
    """

    __slots__ = ("_client", "_prefix", "_ttl_seconds")

    KEY_PREFIX = "icda:result:"

    def __init__(
        self,
        client: RedisStackClient,
        ttl_seconds: int = 3600,
    ):
        """Initialize query result store.

        Args:
            client: Redis Stack client.
            ttl_seconds: Result TTL.
        """
        self._client = client
        self._prefix = self.KEY_PREFIX
        self._ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        """Check if ReJSON is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.JSON)

    def _key(self, query_hash: str) -> str:
        """Generate key for a query result."""
        return f"{self._prefix}{query_hash}"

    async def save(
        self,
        query_hash: str,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Save a query result.

        Args:
            query_hash: Query hash.
            result: Result data.
            ttl: Optional custom TTL.

        Returns:
            True if saved.
        """
        if not self.enabled:
            return False

        key = self._key(query_hash)
        try:
            data = json.dumps(result)
            await self._client.execute("JSON.SET", key, "$", data)
            await self._client.execute("EXPIRE", key, ttl or self._ttl_seconds)
            return True
        except Exception as e:
            print(f"QueryResultStore: Failed to save result: {e}")
            return False

    async def get(self, query_hash: str) -> dict[str, Any] | None:
        """Get a query result.

        Args:
            query_hash: Query hash.

        Returns:
            Result data or None.
        """
        if not self.enabled:
            return None

        key = self._key(query_hash)
        try:
            result = await self._client.execute("JSON.GET", key)
            if result:
                return json.loads(result)
        except Exception as e:
            print(f"QueryResultStore: Failed to get result: {e}")
        return None
