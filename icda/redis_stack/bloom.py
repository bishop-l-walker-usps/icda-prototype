"""Bloom Filter Operations - Probabilistic data structures.

Provides:
- Duplicate query detection
- Rate limiting with counting bloom filters
- Membership testing for large sets
- Novelty detection for cache optimization
"""

from __future__ import annotations

from enum import Enum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import RedisStackClient


class BloomFilterType(str, Enum):
    """Types of bloom filters used in the system."""
    SEEN_QUERIES = "seen_queries"
    VALID_CRIDS = "valid_crids"
    BLOCKED_QUERIES = "blocked_queries"
    HOT_CUSTOMERS = "hot_customers"


class BloomFilters:
    """Redis Bloom filter operations.

    Uses BF.* commands for:
    - Fast duplicate detection (O(1))
    - Memory-efficient membership testing
    - Rate limiting with count-min sketch
    """

    __slots__ = ("_client", "_prefix", "_enabled")

    KEY_PREFIX = "icda:bloom:"

    # Default filter configurations
    FILTER_CONFIGS = {
        BloomFilterType.SEEN_QUERIES: {"capacity": 100000, "error_rate": 0.01},
        BloomFilterType.VALID_CRIDS: {"capacity": 100000, "error_rate": 0.001},
        BloomFilterType.BLOCKED_QUERIES: {"capacity": 10000, "error_rate": 0.01},
        BloomFilterType.HOT_CUSTOMERS: {"capacity": 10000, "error_rate": 0.01},
    }

    def __init__(self, client: RedisStackClient):
        """Initialize bloom filter operations.

        Args:
            client: Redis Stack client.
        """
        self._client = client
        self._prefix = self.KEY_PREFIX

    @property
    def enabled(self) -> bool:
        """Check if Bloom module is available."""
        from .client import RedisModule
        return self._client.has_module(RedisModule.BLOOM)

    def _key(self, filter_type: BloomFilterType) -> str:
        """Generate key for a bloom filter."""
        return f"{self._prefix}{filter_type.value}"

    async def ensure_filter(self, filter_type: BloomFilterType) -> bool:
        """Ensure a bloom filter exists with proper configuration.

        Args:
            filter_type: Filter type to create.

        Returns:
            True if filter exists or was created.
        """
        if not self.enabled:
            return False

        key = self._key(filter_type)
        try:
            # Check if filter exists
            if await self._client.exists(key):
                return True

            # Create with configuration
            config = self.FILTER_CONFIGS.get(filter_type, {"capacity": 10000, "error_rate": 0.01})
            await self._client.execute(
                "BF.RESERVE", key,
                config["error_rate"],
                config["capacity"],
            )
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            print(f"BloomFilters: Failed to create filter {filter_type.value}: {e}")
            return False

    async def add(self, filter_type: BloomFilterType, item: str) -> bool:
        """Add an item to a bloom filter.

        Args:
            filter_type: Filter to add to.
            item: Item to add.

        Returns:
            True if item was added (new), False if might already exist.
        """
        if not self.enabled:
            return False

        key = self._key(filter_type)
        try:
            await self.ensure_filter(filter_type)
            result = await self._client.execute("BF.ADD", key, item)
            return result == 1
        except Exception as e:
            print(f"BloomFilters: Failed to add to {filter_type.value}: {e}")
            return False

    async def exists(self, filter_type: BloomFilterType, item: str) -> bool:
        """Check if an item might exist in a bloom filter.

        Args:
            filter_type: Filter to check.
            item: Item to check.

        Returns:
            True if item might exist, False if definitely doesn't.
        """
        if not self.enabled:
            return False

        key = self._key(filter_type)
        try:
            result = await self._client.execute("BF.EXISTS", key, item)
            return result == 1
        except Exception as e:
            print(f"BloomFilters: Failed to check {filter_type.value}: {e}")
            return False

    async def add_many(self, filter_type: BloomFilterType, items: list[str]) -> list[bool]:
        """Add multiple items to a bloom filter.

        Args:
            filter_type: Filter to add to.
            items: Items to add.

        Returns:
            List of booleans indicating if each item was new.
        """
        if not self.enabled or not items:
            return [False] * len(items)

        key = self._key(filter_type)
        try:
            await self.ensure_filter(filter_type)
            result = await self._client.execute("BF.MADD", key, *items)
            return [r == 1 for r in result]
        except Exception as e:
            print(f"BloomFilters: Failed to add many to {filter_type.value}: {e}")
            return [False] * len(items)

    async def exists_many(self, filter_type: BloomFilterType, items: list[str]) -> list[bool]:
        """Check if multiple items might exist.

        Args:
            filter_type: Filter to check.
            items: Items to check.

        Returns:
            List of booleans for each item.
        """
        if not self.enabled or not items:
            return [False] * len(items)

        key = self._key(filter_type)
        try:
            result = await self._client.execute("BF.MEXISTS", key, *items)
            return [r == 1 for r in result]
        except Exception as e:
            print(f"BloomFilters: Failed to check many in {filter_type.value}: {e}")
            return [False] * len(items)

    async def is_duplicate_query(self, query_hash: str) -> bool:
        """Check if a query has been seen before.

        Args:
            query_hash: Hash of the query.

        Returns:
            True if likely seen before.
        """
        return await self.exists(BloomFilterType.SEEN_QUERIES, query_hash)

    async def mark_query_seen(self, query_hash: str) -> bool:
        """Mark a query as seen.

        Args:
            query_hash: Hash of the query.

        Returns:
            True if this is a new query.
        """
        return await self.add(BloomFilterType.SEEN_QUERIES, query_hash)

    async def is_valid_crid(self, crid: str) -> bool:
        """Check if a CRID is valid.

        Args:
            crid: Customer ID to check.

        Returns:
            True if might be valid.
        """
        return await self.exists(BloomFilterType.VALID_CRIDS, crid)

    async def register_crids(self, crids: list[str]) -> int:
        """Register valid CRIDs.

        Args:
            crids: List of CRIDs to register.

        Returns:
            Number of new CRIDs registered.
        """
        results = await self.add_many(BloomFilterType.VALID_CRIDS, crids)
        return sum(1 for r in results if r)

    async def check_rate_limit(
        self,
        session_id: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """Check rate limit for a session.

        Uses a simple counter with TTL for rate limiting.

        Args:
            session_id: Session to check.
            max_requests: Maximum requests per window.
            window_seconds: Window size in seconds.

        Returns:
            Tuple of (allowed, current_count).
        """
        if not self._client.available:
            return True, 0

        key = f"icda:ratelimit:{session_id}"
        try:
            # Get current count
            count_str = await self._client.client.get(key)
            current = int(count_str) if count_str else 0

            if current >= max_requests:
                return False, current

            # Increment with TTL
            new_count = await self._client.execute("INCR", key)
            if new_count == 1:
                await self._client.execute("EXPIRE", key, window_seconds)

            return True, new_count
        except Exception as e:
            print(f"BloomFilters: Rate limit check failed: {e}")
            return True, 0

    async def get_filter_stats(self, filter_type: BloomFilterType) -> dict[str, Any]:
        """Get statistics for a bloom filter.

        Args:
            filter_type: Filter to get stats for.

        Returns:
            Dict with filter statistics.
        """
        if not self.enabled:
            return {}

        key = self._key(filter_type)
        try:
            info = await self._client.execute("BF.INFO", key)
            if info:
                # Parse info result (alternating keys and values)
                stats = {}
                for i in range(0, len(info), 2):
                    stats[info[i].lower()] = info[i + 1]
                return stats
        except Exception as e:
            print(f"BloomFilters: Failed to get stats for {filter_type.value}: {e}")
        return {}

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all bloom filters.

        Returns:
            Dict mapping filter type to stats.
        """
        stats = {}
        for filter_type in BloomFilterType:
            filter_stats = await self.get_filter_stats(filter_type)
            if filter_stats:
                stats[filter_type.value] = filter_stats
        return stats
