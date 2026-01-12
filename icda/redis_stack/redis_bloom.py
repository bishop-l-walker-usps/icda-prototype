"""RedisBloom wrapper for probabilistic data structures.

Provides:
- Bloom Filter: Query deduplication
- Cuckoo Filter: Document deduplication (supports deletion)
- Count-Min Sketch: Query frequency estimation
- Top-K: Most frequent queries
- HyperLogLog: Unique visitor counting
"""

import asyncio
import logging
from hashlib import sha256

logger = logging.getLogger(__name__)


class RedisBloomWrapper:
    """Wrapper for RedisBloom probabilistic data structures.

    Key naming convention:
        bf:seen_queries        - Bloom filter for query deduplication
        cf:doc_chunks          - Cuckoo filter for document chunks
        cms:query_freq         - Count-Min Sketch for query frequency
        topk:queries:1h        - Top-K trending queries (hourly)
        hll:users:daily        - HyperLogLog for unique users
    """

    # Bloom filter settings
    BLOOM_ERROR_RATE = 0.01  # 1% false positive rate
    BLOOM_CAPACITY = 100000  # Expected items

    # Top-K settings
    TOPK_SIZE = 100
    TOPK_WIDTH = 2000
    TOPK_DEPTH = 7
    TOPK_DECAY = 0.9

    # Count-Min Sketch settings
    CMS_WIDTH = 2000
    CMS_DEPTH = 5

    def __init__(self, redis):
        self.redis = redis

    async def ensure_filters(self) -> None:
        """Create all probabilistic data structures."""
        await self._ensure_bloom("bf:seen_queries", self.BLOOM_CAPACITY, self.BLOOM_ERROR_RATE)
        await self._ensure_cuckoo("cf:doc_chunks", self.BLOOM_CAPACITY)
        await self._ensure_cms("cms:query_freq", self.CMS_WIDTH, self.CMS_DEPTH)
        await self._ensure_topk("topk:queries:1h", self.TOPK_SIZE, self.TOPK_WIDTH, self.TOPK_DEPTH, self.TOPK_DECAY)

    async def _ensure_bloom(self, key: str, capacity: int, error_rate: float) -> bool:
        """Create Bloom filter if not exists."""
        try:
            await self.redis.execute_command("BF.INFO", key)
            return True
        except Exception:
            pass

        try:
            await self.redis.execute_command(
                "BF.RESERVE", key, error_rate, capacity
            )
            logger.debug(f"Created Bloom filter: {key}")
            return True
        except Exception as e:
            if "exists" not in str(e).lower():
                logger.warning(f"Failed to create Bloom filter {key}: {e}")
            return False

    async def _ensure_cuckoo(self, key: str, capacity: int) -> bool:
        """Create Cuckoo filter if not exists."""
        try:
            await self.redis.execute_command("CF.INFO", key)
            return True
        except Exception:
            pass

        try:
            await self.redis.execute_command(
                "CF.RESERVE", key, capacity
            )
            logger.debug(f"Created Cuckoo filter: {key}")
            return True
        except Exception as e:
            if "exists" not in str(e).lower():
                logger.warning(f"Failed to create Cuckoo filter {key}: {e}")
            return False

    async def _ensure_cms(self, key: str, width: int, depth: int) -> bool:
        """Create Count-Min Sketch if not exists."""
        try:
            await self.redis.execute_command("CMS.INFO", key)
            return True
        except Exception:
            pass

        try:
            await self.redis.execute_command(
                "CMS.INITBYDIM", key, width, depth
            )
            logger.debug(f"Created Count-Min Sketch: {key}")
            return True
        except Exception as e:
            if "exists" not in str(e).lower():
                logger.warning(f"Failed to create CMS {key}: {e}")
            return False

    async def _ensure_topk(
        self,
        key: str,
        k: int,
        width: int,
        depth: int,
        decay: float
    ) -> bool:
        """Create Top-K if not exists."""
        try:
            await self.redis.execute_command("TOPK.INFO", key)
            return True
        except Exception:
            pass

        try:
            await self.redis.execute_command(
                "TOPK.RESERVE", key, k, width, depth, decay
            )
            logger.debug(f"Created Top-K: {key}")
            return True
        except Exception as e:
            if "exists" not in str(e).lower():
                logger.warning(f"Failed to create Top-K {key}: {e}")
            return False

    # =========================================================================
    # Query Deduplication (Bloom Filter)
    # =========================================================================

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        return query.casefold().strip()

    async def query_seen(self, query: str) -> bool:
        """Check if query was recently seen.

        Args:
            query: Query string to check

        Returns:
            True if query likely seen before, False if definitely not seen
        """
        try:
            normalized = self._normalize_query(query)
            result = await self.redis.execute_command(
                "BF.EXISTS", "bf:seen_queries", normalized
            )
            return bool(result)
        except Exception as e:
            logger.debug(f"Bloom filter check failed: {e}")
            return False  # Conservative: don't skip if check fails

    async def add_query(self, query: str) -> None:
        """Mark query as seen."""
        try:
            normalized = self._normalize_query(query)
            await self.redis.execute_command(
                "BF.ADD", "bf:seen_queries", normalized
            )
        except Exception as e:
            logger.debug(f"Bloom filter add failed: {e}")

    # =========================================================================
    # Document Deduplication (Cuckoo Filter)
    # =========================================================================

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication."""
        return sha256(content.encode()).hexdigest()[:32]

    async def doc_chunk_seen(self, content: str) -> bool:
        """Check if document chunk already indexed."""
        try:
            content_hash = self._hash_content(content)
            result = await self.redis.execute_command(
                "CF.EXISTS", "cf:doc_chunks", content_hash
            )
            return bool(result)
        except Exception as e:
            logger.debug(f"Cuckoo filter check failed: {e}")
            return False

    async def add_doc_chunk(self, content: str) -> None:
        """Mark document chunk as indexed."""
        try:
            content_hash = self._hash_content(content)
            await self.redis.execute_command(
                "CF.ADD", "cf:doc_chunks", content_hash
            )
        except Exception as e:
            logger.debug(f"Cuckoo filter add failed: {e}")

    async def remove_doc_chunk(self, content: str) -> bool:
        """Remove document chunk from index (Cuckoo supports deletion)."""
        try:
            content_hash = self._hash_content(content)
            result = await self.redis.execute_command(
                "CF.DEL", "cf:doc_chunks", content_hash
            )
            return bool(result)
        except Exception as e:
            logger.debug(f"Cuckoo filter delete failed: {e}")
            return False

    # =========================================================================
    # Query Frequency (Count-Min Sketch + Top-K)
    # =========================================================================

    async def track_query_frequency(self, query: str) -> None:
        """Track query for frequency analysis."""
        try:
            normalized = self._normalize_query(query)

            # Update Count-Min Sketch
            await self.redis.execute_command(
                "CMS.INCRBY", "cms:query_freq", normalized, 1
            )

            # Update Top-K
            await self.redis.execute_command(
                "TOPK.ADD", "topk:queries:1h", normalized
            )
        except Exception as e:
            logger.debug(f"Query frequency tracking failed: {e}")

    async def get_query_frequency(self, query: str) -> int:
        """Get estimated frequency of a query."""
        try:
            normalized = self._normalize_query(query)
            result = await self.redis.execute_command(
                "CMS.QUERY", "cms:query_freq", normalized
            )
            return int(result[0]) if result else 0
        except Exception as e:
            logger.debug(f"Query frequency lookup failed: {e}")
            return 0

    async def get_top_queries(self, k: int = 10) -> list[tuple[str, int]]:
        """Get top K most frequent queries.

        Returns:
            List of (query, count) tuples
        """
        try:
            # TOPK.LIST returns items with counts
            result = await self.redis.execute_command(
                "TOPK.LIST", "topk:queries:1h", "WITHCOUNT"
            )

            if not result:
                return []

            # Parse result: [item1, count1, item2, count2, ...]
            queries = []
            for i in range(0, min(len(result), k * 2), 2):
                if i + 1 < len(result):
                    query = result[i]
                    count = int(result[i + 1]) if result[i + 1] else 0
                    if query:  # Skip None entries
                        queries.append((query, count))

            return queries[:k]
        except Exception as e:
            logger.debug(f"Top queries lookup failed: {e}")
            return []

    # =========================================================================
    # Unique Visitors (HyperLogLog)
    # =========================================================================

    async def count_unique_session(self, session_id: str, key: str = "hll:users:daily") -> None:
        """Add session to unique visitor count."""
        try:
            await self.redis.execute_command("PFADD", key, session_id)
        except Exception as e:
            logger.debug(f"HyperLogLog add failed: {e}")

    async def get_unique_count(self, key: str = "hll:users:daily") -> int:
        """Get approximate unique visitor count."""
        try:
            result = await self.redis.execute_command("PFCOUNT", key)
            return int(result) if result else 0
        except Exception as e:
            logger.debug(f"HyperLogLog count failed: {e}")
            return 0

    # =========================================================================
    # Maintenance
    # =========================================================================

    async def reset_hourly_topk(self) -> None:
        """Reset hourly Top-K (call from scheduled task)."""
        try:
            await self.redis.delete("topk:queries:1h")
            await self._ensure_topk(
                "topk:queries:1h",
                self.TOPK_SIZE,
                self.TOPK_WIDTH,
                self.TOPK_DEPTH,
                self.TOPK_DECAY
            )
        except Exception as e:
            logger.warning(f"Failed to reset hourly Top-K: {e}")

    async def get_filter_stats(self) -> dict:
        """Get stats for all filters."""
        stats = {}

        try:
            bf_info = await self.redis.execute_command("BF.INFO", "bf:seen_queries")
            stats["bloom_filter"] = dict(zip(bf_info[::2], bf_info[1::2])) if bf_info else {}
        except Exception:
            stats["bloom_filter"] = {"error": "unavailable"}

        try:
            cf_info = await self.redis.execute_command("CF.INFO", "cf:doc_chunks")
            stats["cuckoo_filter"] = dict(zip(cf_info[::2], cf_info[1::2])) if cf_info else {}
        except Exception:
            stats["cuckoo_filter"] = {"error": "unavailable"}

        try:
            cms_info = await self.redis.execute_command("CMS.INFO", "cms:query_freq")
            stats["count_min_sketch"] = dict(zip(cms_info[::2], cms_info[1::2])) if cms_info else {}
        except Exception:
            stats["count_min_sketch"] = {"error": "unavailable"}

        return stats
