"""Progress tracking for ingestion pipeline.

Emits events for monitoring and downstream consumers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from icda.ingestion.pipeline.ingestion_models import (
    IngestionEvent,
    IngestionEventData,
)

logger = logging.getLogger(__name__)


EventCallback = Callable[[IngestionEventData], None]


class ProgressTracker:
    """Tracks progress and emits events during ingestion.

    Features:
    - Event emission to registered listeners
    - Progress calculation
    - Statistics tracking
    - Logging integration
    """

    __slots__ = (
        "_batch_id",
        "_total_records",
        "_processed",
        "_approved",
        "_rejected",
        "_errors",
        "_listeners",
        "_emit_interval",
        "_last_emit",
    )

    def __init__(
        self,
        emit_interval: int = 100,
    ):
        """Initialize progress tracker.

        Args:
            emit_interval: Emit progress every N records.
        """
        self._batch_id: str | None = None
        self._total_records = 0
        self._processed = 0
        self._approved = 0
        self._rejected = 0
        self._errors = 0
        self._listeners: list[EventCallback] = []
        self._emit_interval = emit_interval
        self._last_emit = 0

    def add_listener(self, callback: EventCallback) -> None:
        """Register event listener.

        Args:
            callback: Function to call with event data.
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: EventCallback) -> None:
        """Remove event listener.

        Args:
            callback: Listener to remove.
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def start_batch(
        self,
        batch_id: str,
        total_records: int,
    ) -> None:
        """Start tracking a new batch.

        Args:
            batch_id: Batch identifier.
            total_records: Total records in batch.
        """
        self._batch_id = batch_id
        self._total_records = total_records
        self._processed = 0
        self._approved = 0
        self._rejected = 0
        self._errors = 0
        self._last_emit = 0

        self._emit(
            IngestionEvent.BATCH_STARTED,
            details={
                "total_records": total_records,
            },
        )

        logger.info(f"Started batch {batch_id} with {total_records} records")

    def record_processed(
        self,
        record_id: str,
        approved: bool,
        is_duplicate: bool = False,
    ) -> None:
        """Record that a record was processed.

        Args:
            record_id: Record identifier.
            approved: Whether record was approved.
            is_duplicate: Whether record was a duplicate.
        """
        self._processed += 1

        if approved:
            self._approved += 1
            self._emit(
                IngestionEvent.RECORD_APPROVED,
                record_id=record_id,
            )
        elif is_duplicate:
            self._rejected += 1
            self._emit(
                IngestionEvent.RECORD_DUPLICATE,
                record_id=record_id,
            )
        else:
            self._rejected += 1
            self._emit(
                IngestionEvent.RECORD_REJECTED,
                record_id=record_id,
            )

        # Emit progress at intervals
        if self._processed - self._last_emit >= self._emit_interval:
            self._emit_progress()
            self._last_emit = self._processed

    def record_embedding(
        self,
        record_id: str,
        precomputed: bool,
        provider: str | None = None,
    ) -> None:
        """Record embedding event.

        Args:
            record_id: Record identifier.
            precomputed: Whether embedding was pre-computed.
            provider: Embedding provider used.
        """
        event = (
            IngestionEvent.EMBEDDING_PRECOMPUTED
            if precomputed
            else IngestionEvent.EMBEDDING_GENERATED
        )

        self._emit(
            event,
            record_id=record_id,
            details={"provider": provider},
        )

    def record_error(
        self,
        record_id: str | None,
        error: str,
    ) -> None:
        """Record an error.

        Args:
            record_id: Record identifier (if applicable).
            error: Error message.
        """
        self._errors += 1
        self._processed += 1

        self._emit(
            IngestionEvent.ERROR,
            record_id=record_id,
            details={"error": error},
        )

        logger.error(f"Ingestion error for {record_id}: {error}")

    def record_indexed(
        self,
        record_id: str,
        index_type: str,
    ) -> None:
        """Record that a record was indexed.

        Args:
            record_id: Record identifier.
            index_type: Type of index (address, vector).
        """
        self._emit(
            IngestionEvent.INDEX_UPDATED,
            record_id=record_id,
            details={"index_type": index_type},
        )

    def complete_batch(self) -> None:
        """Complete the current batch."""
        self._emit(
            IngestionEvent.BATCH_COMPLETED,
            details={
                "total": self._total_records,
                "processed": self._processed,
                "approved": self._approved,
                "rejected": self._rejected,
                "errors": self._errors,
                "approval_rate": self.approval_rate,
            },
        )

        logger.info(
            f"Completed batch {self._batch_id}: "
            f"{self._approved}/{self._processed} approved "
            f"({self.approval_rate:.1%})"
        )

    @property
    def progress(self) -> float:
        """Get progress as fraction (0.0-1.0)."""
        if self._total_records == 0:
            return 0.0
        return self._processed / self._total_records

    @property
    def approval_rate(self) -> float:
        """Get approval rate."""
        if self._processed == 0:
            return 0.0
        return self._approved / self._processed

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "batch_id": self._batch_id,
            "total_records": self._total_records,
            "processed": self._processed,
            "approved": self._approved,
            "rejected": self._rejected,
            "errors": self._errors,
            "progress": self.progress,
            "approval_rate": self.approval_rate,
        }

    def _emit(
        self,
        event: IngestionEvent,
        record_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit event to all listeners.

        Args:
            event: Event type.
            record_id: Record identifier.
            details: Additional details.
        """
        event_data = IngestionEventData(
            event=event,
            batch_id=self._batch_id,
            record_id=record_id,
            progress=self.progress,
            records_processed=self._processed,
            records_total=self._total_records,
            details=details or {},
        )

        for listener in self._listeners:
            try:
                listener(event_data)
            except Exception as e:
                logger.warning(f"Event listener error: {e}")

    def _emit_progress(self) -> None:
        """Emit progress event."""
        self._emit(
            IngestionEvent.BATCH_PROGRESS,
            details={
                "processed": self._processed,
                "approved": self._approved,
                "rejected": self._rejected,
            },
        )

        logger.debug(
            f"Batch {self._batch_id} progress: "
            f"{self._processed}/{self._total_records} "
            f"({self.progress:.1%})"
        )
