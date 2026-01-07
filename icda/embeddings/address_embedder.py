"""Titan embeddings optimized for address similarity with Redis caching."""

import asyncio
import hashlib
import json
import logging
from typing import List, Optional, Union

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

# Street type expansions for better semantic matching
STREET_TYPE_EXPANSIONS = {
    " st ": " street ", " st,": " street,", " st.": " street ",
    " ave ": " avenue ", " ave,": " avenue,", " ave.": " avenue ",
    " blvd ": " boulevard ", " blvd,": " boulevard,", " blvd.": " boulevard ",
    " ln ": " lane ", " ln,": " lane,", " ln.": " lane ",
    " dr ": " drive ", " dr,": " drive,", " dr.": " drive ",
    " rd ": " road ", " rd,": " road,", " rd.": " road ",
    " ct ": " court ", " ct,": " court,", " ct.": " court ",
    " pl ": " place ", " pl,": " place,", " pl.": " place ",
    " cir ": " circle ", " cir,": " circle,", " cir.": " circle ",
    " ter ": " terrace ", " ter,": " terrace,", " ter.": " terrace ",
    " pkwy ": " parkway ", " pkwy,": " parkway,", " pkwy.": " parkway ",
    " hwy ": " highway ", " hwy,": " highway,", " hwy.": " highway ",
    " apt ": " apartment ", " apt,": " apartment,", " apt.": " apartment ",
    " ste ": " suite ", " ste,": " suite,", " ste.": " suite ",
    " # ": " unit ", "#": " unit ",
}

# Unit type normalizations
UNIT_TYPE_EXPANSIONS = {
    "apt": "apartment",
    "ste": "suite",
    "unit": "unit",
    "fl": "floor",
    "flr": "floor",
    "rm": "room",
    "bldg": "building",
}


class AddressEmbedder:
    """Titan embeddings optimized for address similarity with caching."""

    def __init__(
        self,
        redis_client=None,
        region: str = "us-east-1",
        model: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        cache_ttl: int = 86400 * 7,  # 7 days
    ):
        """Initialize embedder with optional Redis caching.

        Args:
            redis_client: Optional async Redis client for caching
            region: AWS region for Bedrock
            model: Titan embedding model ID
            dimensions: Embedding dimensions
            cache_ttl: Cache TTL in seconds
        """
        self.redis = redis_client
        self.model = model
        self.dimension = dimensions
        self.cache_ttl = cache_ttl
        self.client = None
        self.available = False
        self._fallback_cache: dict = {}  # In-memory fallback

        # Initialize Bedrock client
        try:
            boto_config = BotoConfig(
                read_timeout=60,
                connect_timeout=30,
                retries={"max_attempts": 3},
            )
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=boto_config
            )
            # Verify credentials
            session = boto3.Session()
            if session.get_credentials() is not None:
                self.available = True
                logger.info(f"AddressEmbedder: Titan connected ({model})")
            else:
                logger.warning("AddressEmbedder: No AWS credentials - LITE MODE")
        except NoCredentialsError:
            logger.warning("AddressEmbedder: AWS credentials not found - LITE MODE")
        except Exception as e:
            logger.error(f"AddressEmbedder: Init failed - {e}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        normalized = text.lower().strip()
        return f"emb:addr:{hashlib.md5(normalized.encode()).hexdigest()}"

    def _preprocess_address(self, address: str) -> str:
        """Normalize address for better embedding similarity.

        Expands abbreviations so semantic search works better.
        "123 Main St" -> "123 main street" (more semantic meaning)
        """
        # Lowercase and add padding for word boundary matching
        normalized = f" {address.lower()} "

        # Expand street type abbreviations
        for abbr, full in STREET_TYPE_EXPANSIONS.items():
            normalized = normalized.replace(abbr, full)

        # Clean up extra whitespace
        normalized = " ".join(normalized.split())

        return normalized.strip()

    async def _get_cached(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache (Redis or in-memory)."""
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Redis cache get error: {e}")

        # In-memory fallback
        if key in self._fallback_cache:
            return self._fallback_cache[key]

        return None

    async def _set_cached(self, key: str, embedding: List[float]) -> None:
        """Store embedding in cache (Redis or in-memory)."""
        if self.redis:
            try:
                await self.redis.setex(key, self.cache_ttl, json.dumps(embedding))
                return
            except Exception as e:
                logger.debug(f"Redis cache set error: {e}")

        # In-memory fallback (limited size)
        if len(self._fallback_cache) < 10000:
            self._fallback_cache[key] = embedding

    async def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Get embedding for address text.

        Args:
            text: Address text to embed
            use_cache: Whether to use caching

        Returns:
            Embedding vector or empty list if unavailable
        """
        if not self.available or not self.client:
            return []

        cache_key = self._cache_key(text)

        # Check cache first
        if use_cache:
            cached = await self._get_cached(cache_key)
            if cached:
                return cached

        # Preprocess for better semantic matching
        processed = self._preprocess_address(text)

        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "inputText": processed,
                    "dimensions": self.dimension,
                    "normalize": True
                })
            )

            embedding = json.loads(response["body"].read())["embedding"]

            # Cache result
            if use_cache:
                await self._set_cached(cache_key, embedding)

            return embedding

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("AccessDeniedException", "UnrecognizedClientException"):
                logger.error("AddressEmbedder: AWS access denied - check IAM permissions")
                self.available = False
            return []
        except Exception as e:
            logger.error(f"AddressEmbedder: Embedding error - {e}")
            return []

    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        max_concurrent: int = 10
    ) -> List[List[float]]:
        """Batch embed with parallel processing.

        Args:
            texts: List of address texts
            use_cache: Whether to use caching
            max_concurrent: Max concurrent API calls

        Returns:
            List of embedding vectors
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cached = await self._get_cached(self._cache_key(text))
                if cached:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(texts)))

        # Embed uncached in parallel with semaphore
        if uncached_indices:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def embed_with_limit(idx: int) -> tuple:
                async with semaphore:
                    emb = await self.embed(texts[idx], use_cache=use_cache)
                    return idx, emb

            tasks = [embed_with_limit(i) for i in uncached_indices]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed:
                if isinstance(result, tuple):
                    idx, emb = result
                    results[idx] = emb
                else:
                    logger.error(f"Batch embed error: {result}")

        # Fill any remaining None with empty lists
        return [r if r is not None else [] for r in results]

    def embed_sync(self, text: str) -> List[float]:
        """Synchronous embedding (for non-async contexts).

        Note: Does not use caching to avoid async issues.
        """
        if not self.available or not self.client:
            return []

        processed = self._preprocess_address(text)

        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "inputText": processed,
                    "dimensions": self.dimension,
                    "normalize": True
                })
            )
            return json.loads(response["body"].read())["embedding"]
        except Exception as e:
            logger.error(f"AddressEmbedder: Sync embed error - {e}")
            return []

    async def clear_cache(self) -> int:
        """Clear embedding cache. Returns count of cleared entries."""
        count = len(self._fallback_cache)
        self._fallback_cache.clear()

        if self.redis:
            try:
                # Clear all embedding keys
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match="emb:addr:*", count=100)
                    if keys:
                        deleted += await self.redis.delete(*keys)
                    if cursor == 0:
                        break
                return deleted
            except Exception as e:
                logger.error(f"Cache clear error: {e}")

        return count
