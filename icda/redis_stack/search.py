"""RediSearch Integration - Full-text and vector similarity search.

Provides:
- Query history indexing with full-text search
- Vector similarity search for semantic matching
- Combined text + vector hybrid search
- Faceted search with filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RedisStackClient


class QueryIntent(str, Enum):
    """Query intent classification."""
    LOOKUP = "lookup"
    SEARCH = "search"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    UNKNOWN = "unknown"


@dataclass
class IndexedQuery:
    """A query indexed for search."""
    query_id: str
    query_text: str
    normalized_text: str
    session_id: str
    intent: QueryIntent
    timestamp: float
    latency_ms: float
    cache_hit: bool
    customer_count: int = 0
    embedding: list[float] = field(default_factory=list)

    def to_hash(self) -> dict[str, Any]:
        """Convert to Redis hash format."""
        return {
            "query_text": self.query_text,
            "normalized_text": self.normalized_text,
            "session_id": self.session_id,
            "intent": self.intent.value,
            "timestamp": str(self.timestamp),
            "latency_ms": str(self.latency_ms),
            "cache_hit": "true" if self.cache_hit else "false",
            "customer_count": self.customer_count,
        }

    @classmethod
    def from_hash(cls, query_id: str, data: dict[str, Any]) -> IndexedQuery:
        """Create from Redis hash data."""
        return cls(
            query_id=query_id,
            query_text=data.get("query_text", ""),
            normalized_text=data.get("normalized_text", ""),
            session_id=data.get("session_id", ""),
            intent=QueryIntent(data.get("intent", "unknown")),
            timestamp=float(data.get("timestamp", 0)),
            latency_ms=float(data.get("latency_ms", 0)),
            cache_hit=data.get("cache_hit", "false") == "true",
            customer_count=int(data.get("customer_count", 0)),
        )


class QuerySearchIndex:
    """RediSearch index for query history.

    Provides full-text search over past queries for:
    - Finding similar queries (cache optimization)
    - Query pattern analysis
    - Session context retrieval
    """

    __slots__ = ("_client", "_index_name", "_prefix", "_enabled")

    INDEX_NAME = "icda:query:idx"
    KEY_PREFIX = "icda:query:"

    def __init__(
        self,
        client: RedisStackClient,
        index_name: str | None = None,
    ):
        """Initialize query search index.

        Args:
            client: Redis Stack client.
            index_name: Custom index name. Uses default if None.
        """
        self._client = client
        self._index_name = index_name or self.INDEX_NAME
        self._prefix = self.KEY_PREFIX

    @property
    def enabled(self) -> bool:
        """Check if RediSearch is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.SEARCH)

    async def create_index(self) -> bool:
        """Create or verify the search index.

        Returns:
            True if index exists or was created.
        """
        if not self.enabled:
            return False

        try:
            # Check if index exists
            await self._client.execute("FT.INFO", self._index_name)
            return True
        except Exception:
            pass

        try:
            # Create index with schema
            await self._client.execute(
                "FT.CREATE", self._index_name,
                "ON", "HASH",
                "PREFIX", "1", self._prefix,
                "SCHEMA",
                "query_text", "TEXT", "WEIGHT", "2.0",
                "normalized_text", "TEXT", "WEIGHT", "1.5",
                "session_id", "TAG",
                "intent", "TAG",
                "timestamp", "NUMERIC", "SORTABLE",
                "latency_ms", "NUMERIC",
                "cache_hit", "TAG",
                "customer_count", "NUMERIC",
            )
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            print(f"QuerySearchIndex: Failed to create index: {e}")
            return False

    async def index_query(self, query: IndexedQuery) -> bool:
        """Index a query for search.

        Args:
            query: Query to index.

        Returns:
            True if indexed successfully.
        """
        if not self.enabled:
            return False

        key = f"{self._prefix}{query.query_id}"
        try:
            await self._client.execute("HSET", key, *sum(query.to_hash().items(), ()))
            return True
        except Exception as e:
            print(f"QuerySearchIndex: Failed to index query: {e}")
            return False

    async def search(
        self,
        text: str,
        session_id: str | None = None,
        intent: QueryIntent | None = None,
        limit: int = 10,
    ) -> list[IndexedQuery]:
        """Search for queries matching criteria.

        Args:
            text: Search text.
            session_id: Filter by session.
            intent: Filter by intent.
            limit: Maximum results.

        Returns:
            List of matching queries.
        """
        if not self.enabled:
            return []

        try:
            # Build query
            query_parts = [text]
            if session_id:
                query_parts.append(f"@session_id:{{{session_id}}}")
            if intent:
                query_parts.append(f"@intent:{{{intent.value}}}")

            query = " ".join(query_parts)

            result = await self._client.execute(
                "FT.SEARCH", self._index_name,
                query,
                "LIMIT", "0", str(limit),
            )

            if not result or result[0] == 0:
                return []

            # Parse results (format: [count, key1, [field, value, ...], key2, ...])
            queries = []
            i = 1
            while i < len(result):
                key = result[i]
                if i + 1 < len(result) and isinstance(result[i + 1], list):
                    fields = result[i + 1]
                    data = {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)}
                    query_id = key.replace(self._prefix, "")
                    queries.append(IndexedQuery.from_hash(query_id, data))
                    i += 2
                else:
                    i += 1

            return queries
        except Exception as e:
            print(f"QuerySearchIndex: Search failed: {e}")
            return []

    async def get_recent(self, limit: int = 10) -> list[IndexedQuery]:
        """Get most recent queries.

        Args:
            limit: Maximum results.

        Returns:
            List of recent queries.
        """
        if not self.enabled:
            return []

        try:
            result = await self._client.execute(
                "FT.SEARCH", self._index_name,
                "*",
                "SORTBY", "timestamp", "DESC",
                "LIMIT", "0", str(limit),
            )

            if not result or result[0] == 0:
                return []

            queries = []
            i = 1
            while i < len(result):
                key = result[i]
                if i + 1 < len(result) and isinstance(result[i + 1], list):
                    fields = result[i + 1]
                    data = {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)}
                    query_id = key.replace(self._prefix, "")
                    queries.append(IndexedQuery.from_hash(query_id, data))
                    i += 2
                else:
                    i += 1

            return queries
        except Exception as e:
            print(f"QuerySearchIndex: Recent query failed: {e}")
            return []


class SimilaritySearch:
    """Vector similarity search for semantic query matching.

    Uses RediSearch's vector indexing for KNN search.
    """

    __slots__ = ("_client", "_index_name", "_prefix", "_dimensions", "_enabled")

    INDEX_NAME = "icda:vector:idx"
    KEY_PREFIX = "icda:vector:"

    def __init__(
        self,
        client: RedisStackClient,
        dimensions: int = 1024,
        index_name: str | None = None,
    ):
        """Initialize vector similarity search.

        Args:
            client: Redis Stack client.
            dimensions: Embedding dimensions.
            index_name: Custom index name.
        """
        self._client = client
        self._dimensions = dimensions
        self._index_name = index_name or self.INDEX_NAME
        self._prefix = self.KEY_PREFIX

    @property
    def enabled(self) -> bool:
        """Check if RediSearch is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.SEARCH)

    async def create_index(self) -> bool:
        """Create or verify the vector index.

        Returns:
            True if index exists or was created.
        """
        if not self.enabled:
            return False

        try:
            # Check if index exists
            await self._client.execute("FT.INFO", self._index_name)
            return True
        except Exception:
            pass

        try:
            # Create vector index
            await self._client.execute(
                "FT.CREATE", self._index_name,
                "ON", "HASH",
                "PREFIX", "1", self._prefix,
                "SCHEMA",
                "text", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self._dimensions),
                "DISTANCE_METRIC", "COSINE",
            )
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            print(f"SimilaritySearch: Failed to create index: {e}")
            return False

    async def add(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a document with embedding.

        Args:
            doc_id: Document ID.
            text: Text content.
            embedding: Vector embedding.
            metadata: Optional metadata.

        Returns:
            True if added successfully.
        """
        if not self.enabled:
            return False

        key = f"{self._prefix}{doc_id}"
        try:
            import struct
            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

            fields = ["text", text, "embedding", embedding_bytes]
            if metadata:
                for k, v in metadata.items():
                    fields.extend([k, str(v)])

            await self._client.execute("HSET", key, *fields)
            return True
        except Exception as e:
            print(f"SimilaritySearch: Failed to add document: {e}")
            return False

    async def search(
        self,
        embedding: list[float],
        k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            embedding: Query embedding.
            k: Number of results.
            filter_expr: Optional filter expression.

        Returns:
            List of matching documents with scores.
        """
        if not self.enabled:
            return []

        try:
            import struct
            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

            query = f"*=>[KNN {k} @embedding $vec AS score]"
            if filter_expr:
                query = f"({filter_expr})=>[KNN {k} @embedding $vec AS score]"

            result = await self._client.execute(
                "FT.SEARCH", self._index_name,
                query,
                "PARAMS", "2", "vec", embedding_bytes,
                "SORTBY", "score",
                "RETURN", "3", "text", "score", "__embedding",
                "DIALECT", "2",
            )

            if not result or result[0] == 0:
                return []

            documents = []
            i = 1
            while i < len(result):
                key = result[i]
                if i + 1 < len(result) and isinstance(result[i + 1], list):
                    fields = result[i + 1]
                    data = {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)}
                    data["id"] = key.replace(self._prefix, "")
                    documents.append(data)
                    i += 2
                else:
                    i += 1

            return documents
        except Exception as e:
            print(f"SimilaritySearch: Search failed: {e}")
            return []
