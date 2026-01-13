"""Schema mapping cache.

Persists learned schema mappings for reuse across sessions.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from icda.ingestion.schema.schema_models import SchemaMapping

logger = logging.getLogger(__name__)


class MappingCache:
    """Cache for schema mappings with persistence.

    Features:
    - In-memory cache for fast lookups
    - File-based persistence
    - TTL-based expiration
    - Usage tracking
    """

    __slots__ = ("_cache_path", "_cache", "_ttl_hours", "_max_entries")

    def __init__(
        self,
        cache_path: str = "./data/schema_cache",
        ttl_hours: int = 720,  # 30 days
        max_entries: int = 1000,
    ):
        """Initialize mapping cache.

        Args:
            cache_path: Directory for cache persistence.
            ttl_hours: Hours before entries expire.
            max_entries: Maximum cache entries.
        """
        self._cache_path = cache_path
        self._cache: dict[str, SchemaMapping] = {}
        self._ttl_hours = ttl_hours
        self._max_entries = max_entries

    async def initialize(self) -> None:
        """Load cache from disk."""
        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path, exist_ok=True)
            return

        index_path = os.path.join(self._cache_path, "index.json")
        if not os.path.exists(index_path):
            return

        try:
            with open(index_path, "r") as f:
                index = json.load(f)

            for source_name, metadata in index.items():
                mapping_path = os.path.join(
                    self._cache_path, f"{self._safe_filename(source_name)}.json"
                )
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        mapping_data = json.load(f)
                        mapping = SchemaMapping.from_dict(mapping_data)

                        # Check if expired
                        if not self._is_expired(mapping):
                            self._cache[source_name] = mapping

            logger.info(f"Loaded {len(self._cache)} schema mappings from cache")

        except Exception as e:
            logger.error(f"Failed to load mapping cache: {e}")

    async def get(self, source_name: str) -> SchemaMapping | None:
        """Get mapping from cache.

        Args:
            source_name: Source identifier.

        Returns:
            SchemaMapping or None if not found/expired.
        """
        mapping = self._cache.get(source_name)

        if mapping is None:
            return None

        # Check expiration
        if self._is_expired(mapping):
            await self.remove(source_name)
            return None

        return mapping

    async def put(
        self,
        source_name: str,
        mapping: SchemaMapping,
    ) -> None:
        """Store mapping in cache.

        Args:
            source_name: Source identifier.
            mapping: SchemaMapping to store.
        """
        # Enforce max entries
        if len(self._cache) >= self._max_entries:
            await self._evict_oldest()

        self._cache[source_name] = mapping

        # Persist to disk
        await self._persist_mapping(source_name, mapping)
        await self._update_index()

    async def remove(self, source_name: str) -> None:
        """Remove mapping from cache.

        Args:
            source_name: Source identifier.
        """
        if source_name in self._cache:
            del self._cache[source_name]

        # Remove file
        mapping_path = os.path.join(
            self._cache_path, f"{self._safe_filename(source_name)}.json"
        )
        if os.path.exists(mapping_path):
            os.remove(mapping_path)

        await self._update_index()

    async def clear(self) -> None:
        """Clear all cached mappings."""
        self._cache.clear()

        # Clear files
        if os.path.exists(self._cache_path):
            for file in os.listdir(self._cache_path):
                if file.endswith(".json"):
                    os.remove(os.path.join(self._cache_path, file))

    def list_sources(self) -> list[str]:
        """List all cached source names."""
        return list(self._cache.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "ttl_hours": self._ttl_hours,
            "sources": self.list_sources(),
        }

    def _is_expired(self, mapping: SchemaMapping) -> bool:
        """Check if mapping has expired."""
        try:
            detected_at = datetime.fromisoformat(mapping.detected_at)
            age_hours = (datetime.utcnow() - detected_at).total_seconds() / 3600
            return age_hours > self._ttl_hours
        except Exception:
            return False

    async def _evict_oldest(self) -> None:
        """Evict oldest/least-used entry."""
        if not self._cache:
            return

        # Find entry with lowest use_count and oldest last_used
        oldest_key = None
        oldest_score = float("inf")

        for name, mapping in self._cache.items():
            # Score based on use_count and recency
            score = mapping.use_count
            if mapping.last_used:
                try:
                    last = datetime.fromisoformat(mapping.last_used)
                    age_days = (datetime.utcnow() - last).days
                    score -= age_days * 10  # Penalize old entries
                except Exception:
                    pass

            if score < oldest_score:
                oldest_score = score
                oldest_key = name

        if oldest_key:
            await self.remove(oldest_key)
            logger.info(f"Evicted schema mapping: {oldest_key}")

    async def _persist_mapping(
        self,
        source_name: str,
        mapping: SchemaMapping,
    ) -> None:
        """Persist mapping to file."""
        try:
            os.makedirs(self._cache_path, exist_ok=True)

            mapping_path = os.path.join(
                self._cache_path, f"{self._safe_filename(source_name)}.json"
            )

            with open(mapping_path, "w") as f:
                json.dump(mapping.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist mapping: {e}")

    async def _update_index(self) -> None:
        """Update cache index file."""
        try:
            index: dict[str, dict[str, Any]] = {}

            for name, mapping in self._cache.items():
                index[name] = {
                    "confidence": mapping.confidence,
                    "use_count": mapping.use_count,
                    "detected_at": mapping.detected_at,
                    "last_used": mapping.last_used,
                }

            index_path = os.path.join(self._cache_path, "index.json")
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update cache index: {e}")

    def _safe_filename(self, name: str) -> str:
        """Convert source name to safe filename."""
        import re
        # Replace unsafe characters
        safe = re.sub(r"[^\w\-_.]", "_", name)
        return safe[:100]  # Limit length
