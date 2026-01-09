"""Redis-based vector index for fast address lookup using Redis Stack.

Requires Redis Stack with RediSearch module for vector similarity search.
Falls back to OpenSearch if Redis vector search is unavailable.
"""

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AddressMatch:
    """Result from address vector search."""
    address: str
    city: str
    state: str
    zip_code: str
    crid: str
    full_address: str
    score: float  # Similarity score 0-1
    urbanization: Optional[str] = None
    is_puerto_rico: bool = False


class RedisAddressIndex:
    """Redis-based vector index for fast address lookup.

    Uses Redis Stack's vector search capabilities (RediSearch) for
    high-performance similarity search with filtering.
    """

    INDEX_NAME = "idx:addresses"
    PREFIX = "addr:"

    def __init__(self, redis_client, embedder: "AddressEmbedder"):
        """Initialize Redis vector index.

        Args:
            redis_client: Async Redis client (redis.asyncio)
            embedder: AddressEmbedder instance for generating vectors
        """
        self.redis = redis_client
        self.embedder = embedder
        self.dimension = embedder.dimension
        self.available = False
        self._index_exists = False

    async def initialize(self, timeout: float = 5.0) -> bool:
        """Check if Redis Stack with vector search is available.

        Args:
            timeout: Maximum time to wait for Redis operations (default 5s)
        """
        if not self.redis:
            logger.warning("RedisAddressIndex: No Redis client provided")
            return False

        try:
            # Check for RediSearch module with timeout protection
            try:
                modules = await asyncio.wait_for(
                    self.redis.module_list(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"RedisAddressIndex: module_list() timed out after {timeout}s")
                return False

            module_names = [m.get("name", "").lower() for m in modules]

            if "search" not in module_names and "ft" not in module_names:
                logger.warning("RedisAddressIndex: RediSearch module not found - install Redis Stack")
                return False

            self.available = True
            logger.info("RedisAddressIndex: Redis Stack vector search available")

            # Check if index exists with timeout protection
            try:
                await asyncio.wait_for(
                    self.redis.execute_command("FT.INFO", self.INDEX_NAME),
                    timeout=timeout
                )
                self._index_exists = True
            except asyncio.TimeoutError:
                logger.warning(f"RedisAddressIndex: FT.INFO timed out after {timeout}s")
                self._index_exists = False
            except Exception:
                self._index_exists = False

            return True

        except Exception as e:
            logger.warning(f"RedisAddressIndex: Init failed - {e}")
            return False

    async def create_index(self, recreate: bool = False) -> bool:
        """Create Redis vector search index.

        Args:
            recreate: If True, drop existing index first

        Returns:
            True if index created successfully
        """
        if not self.available:
            return False

        try:
            if recreate and self._index_exists:
                try:
                    await self.redis.execute_command("FT.DROPINDEX", self.INDEX_NAME)
                    logger.info(f"RedisAddressIndex: Dropped existing index {self.INDEX_NAME}")
                except Exception:
                    pass

            # Create index with schema
            # FT.CREATE idx:addresses ON HASH PREFIX 1 addr:
            #   SCHEMA address TEXT WEIGHT 2.0
            #          city TEXT
            #          state TAG
            #          zip TAG
            #          crid TAG
            #          urbanization TEXT
            #          is_puerto_rico TAG
            #          embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 1024 DISTANCE_METRIC COSINE

            await self.redis.execute_command(
                "FT.CREATE", self.INDEX_NAME,
                "ON", "HASH",
                "PREFIX", "1", self.PREFIX,
                "SCHEMA",
                "address", "TEXT", "WEIGHT", "2.0",
                "city", "TEXT",
                "state", "TAG",
                "zip", "TAG",
                "crid", "TAG",
                "full_address", "TEXT",
                "urbanization", "TEXT",
                "is_puerto_rico", "TAG",
                "embedding", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self.dimension),
                "DISTANCE_METRIC", "COSINE"
            )

            self._index_exists = True
            logger.info(f"RedisAddressIndex: Created index {self.INDEX_NAME}")
            return True

        except Exception as e:
            if "Index already exists" in str(e):
                self._index_exists = True
                return True
            logger.error(f"RedisAddressIndex: Failed to create index - {e}")
            return False

    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """Convert float vector to bytes for Redis storage."""
        return np.array(vector, dtype=np.float32).tobytes()

    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert bytes back to float vector."""
        return np.frombuffer(data, dtype=np.float32).tolist()

    async def index_customer(self, customer: Dict[str, Any]) -> bool:
        """Index a single customer address.

        Args:
            customer: Customer dict with address, city, state, zip, crid

        Returns:
            True if indexed successfully
        """
        if not self.available or not self._index_exists:
            return False

        try:
            # Build full address
            full_address = f"{customer['address']}, {customer['city']}, {customer['state']} {customer['zip']}"

            # Generate embedding
            embedding = await self.embedder.embed(full_address)
            if not embedding:
                logger.warning(f"RedisAddressIndex: No embedding for {customer['crid']}")
                return False

            # Detect Puerto Rico
            is_pr = customer.get("state", "").upper() == "PR"
            if not is_pr and customer.get("zip", ""):
                zip_prefix = customer["zip"][:3]
                is_pr = zip_prefix in ("006", "007", "008", "009")

            # Extract urbanization if present
            urbanization = ""
            addr_upper = customer.get("address", "").upper()
            for prefix in ("URB ", "URBANIZACION ", "URBANIZACIÓN "):
                if addr_upper.startswith(prefix):
                    parts = customer["address"].split(",", 1)
                    if parts:
                        urbanization = parts[0].replace(prefix, "").strip()
                    break

            # Store in Redis
            key = f"{self.PREFIX}{customer['crid']}"
            await self.redis.hset(key, mapping={
                "address": customer.get("address", ""),
                "city": customer.get("city", ""),
                "state": customer.get("state", ""),
                "zip": customer.get("zip", ""),
                "crid": customer.get("crid", ""),
                "full_address": full_address,
                "urbanization": urbanization,
                "is_puerto_rico": "1" if is_pr else "0",
                "embedding": self._vector_to_bytes(embedding)
            })

            return True

        except Exception as e:
            logger.error(f"RedisAddressIndex: Index error for {customer.get('crid')} - {e}")
            return False

    async def index_customers_batch(
        self,
        customers: List[Dict[str, Any]],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, int]:
        """Batch index multiple customers.

        Args:
            customers: List of customer dicts
            batch_size: Number to process in parallel
            progress_callback: Optional callback(indexed, total)

        Returns:
            Dict with indexed, failed, skipped counts
        """
        stats = {"indexed": 0, "failed": 0, "skipped": 0}

        if not self.available:
            stats["skipped"] = len(customers)
            return stats

        # Ensure index exists
        if not self._index_exists:
            await self.create_index()

        total = len(customers)
        for i in range(0, total, batch_size):
            batch = customers[i:i + batch_size]

            # Index in parallel
            tasks = [self.index_customer(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result is True:
                    stats["indexed"] += 1
                elif isinstance(result, Exception):
                    stats["failed"] += 1
                else:
                    stats["failed"] += 1

            if progress_callback:
                progress_callback(stats["indexed"], total)

        logger.info(f"RedisAddressIndex: Indexed {stats['indexed']}/{total} addresses")
        return stats

    async def search(
        self,
        query: str,
        top_k: int = 10,
        zip_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        min_score: float = 0.5
    ) -> List[AddressMatch]:
        """Vector similarity search with optional filters.

        Args:
            query: Address query string
            top_k: Number of results to return
            zip_filter: Optional ZIP code filter
            state_filter: Optional state code filter
            min_score: Minimum similarity score (0-1)

        Returns:
            List of AddressMatch results sorted by score
        """
        if not self.available or not self._index_exists:
            return []

        try:
            # Get query embedding
            query_embedding = await self.embedder.embed(query)
            if not query_embedding:
                return []

            query_vector = self._vector_to_bytes(query_embedding)

            # Build filter string
            filters = []
            if zip_filter:
                filters.append(f"@zip:{{{zip_filter}}}")
            if state_filter:
                filters.append(f"@state:{{{state_filter.upper()}}}")

            filter_str = " ".join(filters) if filters else "*"

            # Execute KNN search
            # FT.SEARCH idx:addresses "(@zip:{22222})=>[KNN 10 @embedding $vec]"
            #   PARAMS 2 vec <binary> SORTBY __score DIALECT 2

            results = await self.redis.execute_command(
                "FT.SEARCH", self.INDEX_NAME,
                f"({filter_str})=>[KNN {top_k} @embedding $vec AS vector_score]",
                "PARAMS", "2", "vec", query_vector,
                "SORTBY", "vector_score",
                "RETURN", "8", "address", "city", "state", "zip", "crid",
                "full_address", "urbanization", "is_puerto_rico",
                "DIALECT", "2"
            )

            # Parse results
            # Results format: [total, doc_id, [field, value, ...], doc_id, ...]
            matches = []
            if results and len(results) > 1:
                total = results[0]
                i = 1
                while i < len(results):
                    doc_id = results[i]
                    i += 1

                    if i >= len(results):
                        break

                    fields = results[i]
                    i += 1

                    # Parse fields into dict
                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            field_dict[fields[j]] = fields[j + 1]

                    # Calculate similarity score (1 - distance for cosine)
                    vector_score = float(field_dict.get("vector_score", 1.0))
                    similarity = 1 - vector_score

                    if similarity >= min_score:
                        matches.append(AddressMatch(
                            address=field_dict.get("address", ""),
                            city=field_dict.get("city", ""),
                            state=field_dict.get("state", ""),
                            zip_code=field_dict.get("zip", ""),
                            crid=field_dict.get("crid", ""),
                            full_address=field_dict.get("full_address", ""),
                            score=similarity,
                            urbanization=field_dict.get("urbanization") or None,
                            is_puerto_rico=field_dict.get("is_puerto_rico") == "1"
                        ))

            return sorted(matches, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"RedisAddressIndex: Search error - {e}")
            return []

    async def search_by_urbanization(
        self,
        urbanization: str,
        zip_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[AddressMatch]:
        """Search by urbanization name (Puerto Rico addresses).

        Args:
            urbanization: Urbanization name to search
            zip_filter: Optional ZIP code filter
            top_k: Max results

        Returns:
            List of matching addresses
        """
        if not self.available or not self._index_exists:
            return []

        try:
            # Text search on urbanization field
            query_parts = [f"@urbanization:{urbanization}"]
            if zip_filter:
                query_parts.append(f"@zip:{{{zip_filter}}}")

            query_str = " ".join(query_parts)

            results = await self.redis.execute_command(
                "FT.SEARCH", self.INDEX_NAME, query_str,
                "RETURN", "8", "address", "city", "state", "zip", "crid",
                "full_address", "urbanization", "is_puerto_rico",
                "LIMIT", "0", str(top_k)
            )

            matches = []
            if results and len(results) > 1:
                i = 1
                while i < len(results):
                    doc_id = results[i]
                    i += 1
                    if i >= len(results):
                        break
                    fields = results[i]
                    i += 1

                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            field_dict[fields[j]] = fields[j + 1]

                    matches.append(AddressMatch(
                        address=field_dict.get("address", ""),
                        city=field_dict.get("city", ""),
                        state=field_dict.get("state", ""),
                        zip_code=field_dict.get("zip", ""),
                        crid=field_dict.get("crid", ""),
                        full_address=field_dict.get("full_address", ""),
                        score=0.9,  # Text match score
                        urbanization=field_dict.get("urbanization") or None,
                        is_puerto_rico=field_dict.get("is_puerto_rico") == "1"
                    ))

            return matches

        except Exception as e:
            logger.error(f"RedisAddressIndex: Urbanization search error - {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.available:
            return {"available": False, "indexed": False}

        try:
            info = await self.redis.execute_command("FT.INFO", self.INDEX_NAME)

            # Parse info response
            info_dict = {}
            for i in range(0, len(info), 2):
                if i + 1 < len(info):
                    info_dict[info[i]] = info[i + 1]

            return {
                "available": True,
                "indexed": self._index_exists,
                "num_docs": info_dict.get("num_docs", 0),
                "num_records": info_dict.get("num_records", 0),
                "index_name": self.INDEX_NAME,
                "dimension": self.dimension
            }

        except Exception as e:
            return {"available": True, "indexed": False, "error": str(e)}

    async def delete_all(self) -> int:
        """Delete all indexed addresses. Returns count deleted."""
        if not self.available:
            return 0

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=f"{self.PREFIX}*", count=100
                )
                if keys:
                    deleted += await self.redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"RedisAddressIndex: Deleted {deleted} addresses")
            return deleted

        except Exception as e:
            logger.error(f"RedisAddressIndex: Delete error - {e}")
            return 0
