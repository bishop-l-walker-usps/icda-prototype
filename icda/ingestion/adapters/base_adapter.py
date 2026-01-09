"""Base adapter interface for data sources.

Abstract base class for all data source adapters with consistent
interface for connecting, reading, and tracking statistics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from icda.ingestion.pipeline.ingestion_models import AddressRecord


class AdapterType(str, Enum):
    """Types of data source adapters."""

    NCOA_BATCH = "ncoa_batch"          # C library NCOA file output
    REST_WEBHOOK = "rest_webhook"      # Real-time REST updates
    FILE_WATCHER = "file_watcher"      # CSV/JSON file drops
    STREAM = "stream"                  # Generic stream adapter
    MEMORY = "memory"                  # In-memory test adapter


@dataclass(slots=True)
class AdapterStats:
    """Statistics for adapter operations."""

    records_read: int = 0
    records_with_embeddings: int = 0
    records_without_embeddings: int = 0
    batches_processed: int = 0
    errors: int = 0
    bytes_read: int = 0
    last_read_time: str | None = None
    last_error: str | None = None

    def record_read(self, has_embedding: bool = False, bytes_count: int = 0) -> None:
        """Record a successful read."""
        self.records_read += 1
        if has_embedding:
            self.records_with_embeddings += 1
        else:
            self.records_without_embeddings += 1
        self.bytes_read += bytes_count
        self.last_read_time = datetime.utcnow().isoformat()

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors += 1
        self.last_error = error
        self.last_read_time = datetime.utcnow().isoformat()

    def record_batch(self) -> None:
        """Record a batch completion."""
        self.batches_processed += 1

    @property
    def embedding_rate(self) -> float:
        """Calculate rate of records with embeddings."""
        if self.records_read == 0:
            return 0.0
        return self.records_with_embeddings / self.records_read

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "records_read": self.records_read,
            "records_with_embeddings": self.records_with_embeddings,
            "records_without_embeddings": self.records_without_embeddings,
            "batches_processed": self.batches_processed,
            "errors": self.errors,
            "bytes_read": self.bytes_read,
            "embedding_rate": round(self.embedding_rate * 100, 2),
            "last_read_time": self.last_read_time,
            "last_error": self.last_error,
        }


class BaseStreamAdapter(ABC):
    """Abstract base class for data source adapters.

    Adapters handle:
    - Connecting to data sources
    - Reading records as async streams
    - Detecting pre-computed embeddings
    - Error handling and retries
    - Checkpoint management for resumability

    Implementations must:
    - connect(): Establish connection
    - disconnect(): Clean up
    - read_stream(): Yield AddressRecords
    - has_precomputed_embeddings(): Report embedding availability
    """

    __slots__ = ("_name", "_config", "_stats", "_enabled", "_connected")

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize adapter.

        Args:
            name: Adapter name for identification.
            config: Configuration dictionary.
        """
        self._name = name
        self._config = config or {}
        self._stats = AdapterStats()
        self._enabled = True
        self._connected = False

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    @abstractmethod
    def adapter_type(self) -> AdapterType:
        """Return adapter type identifier."""
        pass

    @property
    def stats(self) -> AdapterStats:
        """Get adapter statistics."""
        return self._stats

    @property
    def enabled(self) -> bool:
        """Check if adapter is enabled."""
        return self._enabled

    @property
    def connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected

    def enable(self) -> None:
        """Enable the adapter."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the adapter."""
        self._enabled = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection resources."""
        pass

    @abstractmethod
    async def read_stream(self) -> AsyncIterator[AddressRecord]:
        """Yield address records from source.

        Implementations should:
        - Handle pagination/batching internally
        - Detect and include pre-computed embeddings
        - Update stats as records are read
        - Handle errors gracefully

        Yields:
            AddressRecord instances.
        """
        pass

    @abstractmethod
    def has_precomputed_embeddings(self) -> bool:
        """Check if source provides pre-computed embeddings.

        Returns:
            True if embeddings are available from source.
        """
        pass

    async def read_batch(self, batch_size: int = 1000) -> list[AddressRecord]:
        """Read a batch of records.

        Convenience method that collects records from stream.

        Args:
            batch_size: Maximum records to read.

        Returns:
            List of AddressRecords.
        """
        records: list[AddressRecord] = []
        count = 0

        async for record in self.read_stream():
            records.append(record)
            count += 1
            if count >= batch_size:
                break

        self._stats.record_batch()
        return records

    async def count_records(self) -> int:
        """Count total records available.

        Default implementation reads entire stream.
        Override for more efficient counting.

        Returns:
            Total record count.
        """
        count = 0
        async for _ in self.read_stream():
            count += 1
        return count

    def get_info(self) -> dict[str, Any]:
        """Get adapter information."""
        return {
            "name": self._name,
            "type": self.adapter_type.value,
            "enabled": self._enabled,
            "connected": self._connected,
            "has_precomputed_embeddings": self.has_precomputed_embeddings(),
            "stats": self._stats.to_dict(),
        }


class MemoryAdapter(BaseStreamAdapter):
    """In-memory adapter for testing.

    Holds records in memory for testing purposes.
    """

    __slots__ = ("_records",)

    def __init__(self, records: list[AddressRecord] | None = None):
        """Initialize with optional records.

        Args:
            records: Pre-loaded records.
        """
        super().__init__("memory_adapter")
        self._records = records or []

    @property
    def adapter_type(self) -> AdapterType:
        return AdapterType.MEMORY

    async def connect(self) -> bool:
        """Connect - always succeeds for memory adapter."""
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect - no-op for memory adapter."""
        self._connected = False

    async def read_stream(self) -> AsyncIterator[AddressRecord]:
        """Yield records from memory."""
        for record in self._records:
            self._stats.record_read(
                has_embedding=record.has_precomputed_embedding
            )
            yield record

    def has_precomputed_embeddings(self) -> bool:
        """Check if any record has embeddings."""
        return any(r.has_precomputed_embedding for r in self._records)

    def add_record(self, record: AddressRecord) -> None:
        """Add a record to memory."""
        self._records.append(record)

    def add_records(self, records: list[AddressRecord]) -> None:
        """Add multiple records."""
        self._records.extend(records)

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    async def count_records(self) -> int:
        """Count records in memory."""
        return len(self._records)
