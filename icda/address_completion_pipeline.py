"""Unified address completion pipeline using Redis cache, Titan embeddings, and Nova reranking.

Pipeline flow:
1. L1 Cache: Check Redis completion cache for exact/near matches
2. L2 Vector Search: Redis vector similarity search with Titan embeddings
3. L3 Nova Rerank: Use Nova LLM to select best match from candidates

This provides a fast, accurate, and cost-effective address completion system.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class CompletionSource(str, Enum):
    """Source of completion result."""
    CACHE = "cache"
    VECTOR = "vector"
    NOVA = "nova"
    FALLBACK = "fallback"
    NONE = "none"


@dataclass
class CompletionResult:
    """Result from address completion."""
    original: str
    completed: Optional[str]
    confidence: float
    source: CompletionSource
    crid: Optional[str] = None
    suggestions: Optional[List[str]] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["source"] = self.source.value
        return d


@dataclass
class BatchResult:
    """Result from batch completion."""
    results: List[CompletionResult]
    total: int
    completed: int
    failed: int
    from_cache: int
    from_vector: int
    from_nova: int


class AddressCompletionPipeline:
    """Unified address completion pipeline.

    Combines Redis caching, Titan vector embeddings, and Nova reranking
    for fast and accurate address completion.
    """

    def __init__(
        self,
        redis_client=None,
        embedder=None,
        vector_index=None,
        reranker=None,
        cache_ttl: int = 86400,  # 24 hours
        vector_confidence_threshold: float = 0.85,
        min_confidence: float = 0.5
    ):
        """Initialize completion pipeline.

        Args:
            redis_client: Async Redis client
            embedder: AddressEmbedder instance
            vector_index: RedisAddressIndex instance
            reranker: NovaAddressReranker instance
            cache_ttl: Cache TTL in seconds
            vector_confidence_threshold: Skip Nova if vector score >= this
            min_confidence: Minimum confidence to return a result
        """
        self.redis = redis_client
        self.embedder = embedder
        self.vector_index = vector_index
        self.reranker = reranker
        self.cache_ttl = cache_ttl
        self.vector_threshold = vector_confidence_threshold
        self.min_confidence = min_confidence

        # In-memory fallback cache
        self._fallback_cache: Dict[str, Dict] = {}

    @property
    def available(self) -> bool:
        """Check if pipeline is operational."""
        return (
            self.embedder is not None and
            self.embedder.available and
            self.vector_index is not None
        )

    def _cache_key(self, query: str) -> str:
        """Generate cache key for completion result."""
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return f"completion:{hashlib.md5(normalized.encode()).hexdigest()}"

    def _extract_zip(self, query: str) -> Optional[str]:
        """Extract ZIP code from query if present."""
        words = query.split()
        for word in words:
            # Clean word of punctuation
            clean = word.strip(",.;:")
            if clean.isdigit() and len(clean) == 5:
                return clean
            # Handle ZIP+4
            if "-" in clean:
                parts = clean.split("-")
                if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 5:
                    return parts[0]
        return None

    def _extract_state(self, query: str) -> Optional[str]:
        """Extract state code from query if present."""
        # Common state patterns
        words = query.replace(",", " ").split()
        for word in words:
            clean = word.strip(",.;:").upper()
            if len(clean) == 2 and clean.isalpha():
                # Validate it's a real state (basic check)
                if clean in {
                    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI",
                    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
                }:
                    return clean
        return None

    async def _get_cached(self, key: str) -> Optional[Dict]:
        """Get from cache (Redis or in-memory)."""
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Cache get error: {e}")

        # In-memory fallback
        return self._fallback_cache.get(key)

    async def _set_cached(self, key: str, data: Dict) -> None:
        """Store in cache (Redis or in-memory)."""
        if self.redis:
            try:
                await self.redis.setex(key, self.cache_ttl, json.dumps(data))
                return
            except Exception as e:
                logger.debug(f"Cache set error: {e}")

        # In-memory fallback (limited size)
        if len(self._fallback_cache) < 5000:
            self._fallback_cache[key] = data

    async def complete(
        self,
        address: str,
        use_cache: bool = True,
        return_suggestions: bool = False
    ) -> CompletionResult:
        """Complete a partial address.

        Args:
            address: Partial or incomplete address
            use_cache: Whether to use caching
            return_suggestions: Include alternative suggestions

        Returns:
            CompletionResult with completed address
        """
        if not address or not address.strip():
            return CompletionResult(
                original=address,
                completed=None,
                confidence=0.0,
                source=CompletionSource.NONE,
                reason="Empty input"
            )

        address = address.strip()
        cache_key = self._cache_key(address)

        # L1: Check completion cache
        if use_cache:
            cached = await self._get_cached(cache_key)
            if cached:
                return CompletionResult(
                    original=address,
                    completed=cached.get("completed"),
                    confidence=cached.get("confidence", 0.9),
                    source=CompletionSource.CACHE,
                    crid=cached.get("crid")
                )

        # Check if pipeline is available
        if not self.available:
            return CompletionResult(
                original=address,
                completed=None,
                confidence=0.0,
                source=CompletionSource.NONE,
                reason="Pipeline not available (check AWS credentials)"
            )

        # Extract filters for better accuracy
        zip_filter = self._extract_zip(address)
        state_filter = self._extract_state(address)

        # L2: Vector search
        try:
            candidates = await self.vector_index.search(
                query=address,
                top_k=10,
                zip_filter=zip_filter,
                state_filter=state_filter,
                min_score=self.min_confidence
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            candidates = []

        if not candidates:
            return CompletionResult(
                original=address,
                completed=None,
                confidence=0.0,
                source=CompletionSource.NONE,
                reason="No matching addresses found"
            )

        # Convert candidates to dicts for reranker
        candidate_dicts = [
            {
                "full_address": c.full_address,
                "score": c.score,
                "crid": c.crid,
                "address": c.address,
                "city": c.city,
                "state": c.state,
                "zip": c.zip_code
            }
            for c in candidates
        ]

        # High confidence vector match - skip Nova
        if candidates[0].score >= self.vector_threshold:
            result = CompletionResult(
                original=address,
                completed=candidates[0].full_address,
                confidence=candidates[0].score,
                source=CompletionSource.VECTOR,
                crid=candidates[0].crid,
                suggestions=[c.full_address for c in candidates[1:4]] if return_suggestions else None
            )
        else:
            # L3: Nova reranking for ambiguous cases
            if self.reranker and self.reranker.available:
                rerank_result = await self.reranker.rerank(
                    query=address,
                    candidates=candidate_dicts,
                    return_reasoning=True
                )

                result = CompletionResult(
                    original=address,
                    completed=rerank_result.match,
                    confidence=rerank_result.confidence,
                    source=CompletionSource.NOVA,
                    crid=rerank_result.crid,
                    reason=rerank_result.reason,
                    suggestions=[c.full_address for c in candidates[:4]
                                if c.full_address != rerank_result.match] if return_suggestions else None
                )
            else:
                # Fallback to best vector match
                result = CompletionResult(
                    original=address,
                    completed=candidates[0].full_address,
                    confidence=candidates[0].score,
                    source=CompletionSource.FALLBACK,
                    crid=candidates[0].crid,
                    reason="Nova unavailable, using vector similarity",
                    suggestions=[c.full_address for c in candidates[1:4]] if return_suggestions else None
                )

        # Cache successful results
        if use_cache and result.completed and result.confidence >= self.min_confidence:
            await self._set_cached(cache_key, {
                "completed": result.completed,
                "confidence": result.confidence,
                "crid": result.crid
            })

        return result

    async def complete_batch(
        self,
        addresses: List[str],
        concurrency: int = 10
    ) -> BatchResult:
        """Complete multiple addresses in parallel.

        Args:
            addresses: List of addresses to complete
            concurrency: Max concurrent completions

        Returns:
            BatchResult with all results and statistics
        """
        results: List[CompletionResult] = []
        stats = {
            "total": len(addresses),
            "completed": 0,
            "failed": 0,
            "from_cache": 0,
            "from_vector": 0,
            "from_nova": 0
        }

        semaphore = asyncio.Semaphore(concurrency)

        async def complete_with_limit(addr: str) -> CompletionResult:
            async with semaphore:
                return await self.complete(addr)

        # Run in parallel
        tasks = [complete_with_limit(addr) for addr in addresses]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, CompletionResult):
                results.append(result)
                if result.completed:
                    stats["completed"] += 1
                    if result.source == CompletionSource.CACHE:
                        stats["from_cache"] += 1
                    elif result.source == CompletionSource.VECTOR:
                        stats["from_vector"] += 1
                    elif result.source == CompletionSource.NOVA:
                        stats["from_nova"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1
                results.append(CompletionResult(
                    original="",
                    completed=None,
                    confidence=0.0,
                    source=CompletionSource.NONE,
                    reason=f"Error: {str(result)[:100]}"
                ))

        return BatchResult(
            results=results,
            **stats
        )

    async def warmup_cache(
        self,
        customers: List[Dict[str, Any]],
        include_variations: bool = True
    ) -> Dict[str, int]:
        """Pre-populate cache with customer addresses.

        Args:
            customers: List of customer dicts
            include_variations: Also cache common partial patterns

        Returns:
            Stats dict with counts
        """
        stats = {"exact": 0, "variations": 0}

        for customer in customers:
            full = f"{customer['address']}, {customer['city']}, {customer['state']} {customer['zip']}"

            # Cache exact match
            await self._set_cached(
                self._cache_key(full),
                {
                    "completed": full,
                    "confidence": 1.0,
                    "crid": customer.get("crid")
                }
            )
            stats["exact"] += 1

            if include_variations:
                # Common partial patterns
                variations = [
                    f"{customer['address']} {customer['zip']}",
                    f"{customer['address'].lower()} {customer['city'].lower()}",
                    f"{customer['address']} {customer['city']} {customer['state']}",
                    customer['address'].lower(),
                ]

                for partial in variations:
                    await self._set_cached(
                        self._cache_key(partial),
                        {
                            "completed": full,
                            "confidence": 0.95,
                            "crid": customer.get("crid")
                        }
                    )
                    stats["variations"] += 1

        logger.info(f"Cache warmup: {stats['exact']} exact, {stats['variations']} variations")
        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "available": self.available,
            "embedder_available": self.embedder.available if self.embedder else False,
            "reranker_available": self.reranker.available if self.reranker else False,
            "cache_type": "redis" if self.redis else "memory",
            "fallback_cache_size": len(self._fallback_cache),
            "vector_threshold": self.vector_threshold,
            "min_confidence": self.min_confidence
        }

        if self.vector_index:
            try:
                index_stats = await self.vector_index.get_stats()
                stats["vector_index"] = index_stats
            except Exception as e:
                stats["vector_index"] = {"error": str(e)}

        return stats

    async def clear_cache(self) -> int:
        """Clear completion cache. Returns count cleared."""
        count = len(self._fallback_cache)
        self._fallback_cache.clear()

        if self.redis:
            try:
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match="completion:*", count=100)
                    if keys:
                        deleted += await self.redis.delete(*keys)
                    if cursor == 0:
                        break
                return deleted
            except Exception as e:
                logger.error(f"Cache clear error: {e}")

        return count
