"""Ingestion pipeline data models.

Core data structures for the address ingestion system including
records, statuses, batch results, and event tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from icda.address_models import ParsedAddress


class IngestionStatus(str, Enum):
    """Status of an ingestion record through the pipeline."""

    PENDING = "pending"           # Not yet processed
    SCHEMA_MAPPED = "schema_mapped"  # Schema mapping complete
    NORMALIZED = "normalized"     # Address normalized
    DUPLICATE = "duplicate"       # Detected as duplicate
    QUALITY_FLAGGED = "quality_flagged"  # Low quality, needs review
    APPROVED = "approved"         # Passed all gates, ready for index
    REJECTED = "rejected"         # Failed quality gates
    INDEXED = "indexed"           # Successfully added to indexes
    ERROR = "error"               # Processing error


class IngestionEvent(str, Enum):
    """Events emitted during ingestion processing."""

    BATCH_STARTED = "batch_started"
    BATCH_PROGRESS = "batch_progress"
    BATCH_COMPLETED = "batch_completed"
    RECORD_APPROVED = "record_approved"
    RECORD_REJECTED = "record_rejected"
    RECORD_DUPLICATE = "record_duplicate"
    EMBEDDING_GENERATED = "embedding_generated"
    EMBEDDING_PRECOMPUTED = "embedding_precomputed"
    INDEX_UPDATED = "index_updated"
    SCHEMA_DETECTED = "schema_detected"
    ERROR = "error"


@dataclass(slots=True)
class AddressRecord:
    """Raw address record from any data source.

    This is the input format from adapters before any processing.

    Attributes:
        source_id: Unique identifier from the source system.
        raw_data: Original data as dictionary (varies by source).
        raw_address: Concatenated address string if available.
        precomputed_embedding: Pre-computed embedding vector from source.
        source_metadata: Source-specific metadata (file, timestamp, etc.).
        timestamp: When record was received/read.
    """

    source_id: str
    raw_data: dict[str, Any]
    raw_address: str | None = None
    precomputed_embedding: list[float] | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def has_precomputed_embedding(self) -> bool:
        """Check if source provided pre-computed embedding."""
        return self.precomputed_embedding is not None and len(self.precomputed_embedding) > 0

    @property
    def embedding_dimension(self) -> int | None:
        """Get dimension of pre-computed embedding if present."""
        if self.precomputed_embedding:
            return len(self.precomputed_embedding)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "raw_data": self.raw_data,
            "raw_address": self.raw_address,
            "has_precomputed_embedding": self.has_precomputed_embedding,
            "embedding_dimension": self.embedding_dimension,
            "source_metadata": self.source_metadata,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class IngestionRecord:
    """Address record as it moves through the ingestion pipeline.

    Enriched version of AddressRecord with processing state,
    parsed address, embedding, and quality information.

    Attributes:
        source_record: Original AddressRecord from adapter.
        parsed_address: Normalized ParsedAddress after schema mapping.
        embedding: Final embedding vector (pre-computed or generated).
        embedding_provider: Provider that generated/supplied embedding.
        quality_score: Overall quality score from enforcers (0.0-1.0).
        status: Current pipeline status.
        customer_id: Customer ID if matched/assigned.
        enforcer_results: Results from each enforcer stage.
        error_message: Error details if status is ERROR.
        processing_time_ms: Total processing time.
    """

    source_record: AddressRecord
    parsed_address: ParsedAddress | None = None
    embedding: list[float] | None = None
    embedding_provider: str | None = None
    quality_score: float = 0.0
    status: IngestionStatus = IngestionStatus.PENDING
    customer_id: str | None = None
    enforcer_results: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    processing_time_ms: int = 0

    @property
    def source_id(self) -> str:
        """Get source ID from underlying record."""
        return self.source_record.source_id

    @property
    def is_approved(self) -> bool:
        """Check if record passed all quality gates."""
        return self.status in (IngestionStatus.APPROVED, IngestionStatus.INDEXED)

    @property
    def is_rejected(self) -> bool:
        """Check if record was rejected."""
        return self.status in (
            IngestionStatus.REJECTED,
            IngestionStatus.DUPLICATE,
            IngestionStatus.ERROR,
        )

    @property
    def has_embedding(self) -> bool:
        """Check if embedding is available."""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "parsed_address": self.parsed_address.to_dict() if self.parsed_address else None,
            "has_embedding": self.has_embedding,
            "embedding_provider": self.embedding_provider,
            "quality_score": self.quality_score,
            "status": self.status.value,
            "customer_id": self.customer_id,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass(slots=True)
class IngestionEventData:
    """Data payload for ingestion events.

    Used by ProgressTracker to emit events to downstream consumers.
    """

    event: IngestionEvent
    batch_id: str | None = None
    record_id: str | None = None
    progress: float = 0.0  # 0.0 - 1.0
    records_processed: int = 0
    records_total: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event": self.event.value,
            "batch_id": self.batch_id,
            "record_id": self.record_id,
            "progress": self.progress,
            "records_processed": self.records_processed,
            "records_total": self.records_total,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class IngestionBatchSummary:
    """Summary statistics for batch ingestion.

    Aggregates results across all records in a batch.
    """

    total: int = 0
    approved: int = 0
    rejected: int = 0
    duplicates: int = 0
    quality_flagged: int = 0
    errors: int = 0
    embeddings_precomputed: int = 0
    embeddings_generated: int = 0
    total_time_ms: int = 0

    @property
    def avg_time_ms(self) -> float:
        """Calculate average processing time per record."""
        return self.total_time_ms / self.total if self.total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate approval rate."""
        return self.approved / self.total if self.total > 0 else 0.0

    @property
    def precomputed_rate(self) -> float:
        """Calculate rate of pre-computed embeddings used."""
        total_embeddings = self.embeddings_precomputed + self.embeddings_generated
        if total_embeddings == 0:
            return 0.0
        return self.embeddings_precomputed / total_embeddings

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total": self.total,
            "approved": self.approved,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "quality_flagged": self.quality_flagged,
            "errors": self.errors,
            "embeddings_precomputed": self.embeddings_precomputed,
            "embeddings_generated": self.embeddings_generated,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "success_rate": round(self.success_rate * 100, 2),
            "precomputed_rate": round(self.precomputed_rate * 100, 2),
        }


@dataclass(slots=True)
class IngestionBatchResult:
    """Complete result of batch ingestion.

    Contains all processed records and summary statistics.
    """

    batch_id: str
    source_name: str
    records: list[IngestionRecord] = field(default_factory=list)
    summary: IngestionBatchSummary = field(default_factory=IngestionBatchSummary)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved_records(self) -> list[IngestionRecord]:
        """Get only approved records."""
        return [r for r in self.records if r.is_approved]

    @property
    def rejected_records(self) -> list[IngestionRecord]:
        """Get only rejected records."""
        return [r for r in self.records if r.is_rejected]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "batch_id": self.batch_id,
            "source_name": self.source_name,
            "summary": self.summary.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "records_count": len(self.records),
            "metadata": self.metadata,
        }
