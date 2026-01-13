"""Batch processor for concurrent ingestion.

Handles parallel processing of address records with
configurable concurrency and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, TYPE_CHECKING

from icda.ingestion.pipeline.ingestion_models import (
    AddressRecord,
    IngestionRecord,
    IngestionStatus,
    IngestionBatchResult,
    IngestionBatchSummary,
)
from icda.ingestion.pipeline.progress_tracker import ProgressTracker

if TYPE_CHECKING:
    from icda.ingestion.adapters.base_adapter import BaseStreamAdapter
    from icda.ingestion.embeddings.provider_chain import EmbeddingProviderChain
    from icda.ingestion.schema.schema_mapper import AISchemaMapper
    from icda.ingestion.enforcers.ingestion_coordinator import IngestionEnforcerCoordinator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes batches of address records concurrently.

    Features:
    - Configurable concurrency via semaphore
    - Schema mapping per record
    - Embedding generation/passthrough
    - Enforcer pipeline execution
    - Progress tracking
    """

    __slots__ = (
        "_schema_mapper",
        "_embedding_chain",
        "_enforcer_coordinator",
        "_progress_tracker",
        "_max_concurrent",
        "_semaphore",
    )

    def __init__(
        self,
        schema_mapper: AISchemaMapper | None = None,
        embedding_chain: EmbeddingProviderChain | None = None,
        enforcer_coordinator: IngestionEnforcerCoordinator | None = None,
        progress_tracker: ProgressTracker | None = None,
        max_concurrent: int = 10,
    ):
        """Initialize batch processor.

        Args:
            schema_mapper: AI schema mapper for field detection.
            embedding_chain: Embedding provider chain.
            enforcer_coordinator: Enforcer pipeline coordinator.
            progress_tracker: Progress tracking instance.
            max_concurrent: Maximum concurrent record processing.
        """
        self._schema_mapper = schema_mapper
        self._embedding_chain = embedding_chain
        self._enforcer_coordinator = enforcer_coordinator
        self._progress_tracker = progress_tracker or ProgressTracker()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(
        self,
        records: list[AddressRecord],
        source_name: str,
        schema_mapping: Any | None = None,
    ) -> IngestionBatchResult:
        """Process a batch of address records.

        Args:
            records: List of AddressRecords to process.
            source_name: Name of the data source.
            schema_mapping: Optional pre-detected schema mapping.

        Returns:
            IngestionBatchResult with all processed records.
        """
        batch_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Start progress tracking
        self._progress_tracker.start_batch(batch_id, len(records))

        # Reset enforcer batch state
        if self._enforcer_coordinator:
            self._enforcer_coordinator.reset_batch()

        # Detect schema if not provided and mapper available
        if not schema_mapping and self._schema_mapper:
            sample_records = [r.raw_data for r in records[:5]]
            schema_mapping = await self._schema_mapper.detect_mapping(
                source_name=source_name,
                sample_records=sample_records,
            )

        # Process records concurrently
        tasks = [
            self._process_record(record, schema_mapping)
            for record in records
        ]
        processed_records = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results
        results: list[IngestionRecord] = []
        summary = IngestionBatchSummary()

        for i, result in enumerate(processed_records):
            if isinstance(result, Exception):
                # Create error record
                error_record = IngestionRecord(
                    source_record=records[i],
                    status=IngestionStatus.ERROR,
                    error_message=str(result),
                )
                results.append(error_record)
                summary.errors += 1

                self._progress_tracker.record_error(
                    records[i].source_id, str(result)
                )
            else:
                results.append(result)

                # Update summary
                if result.status == IngestionStatus.APPROVED:
                    summary.approved += 1
                elif result.status == IngestionStatus.INDEXED:
                    summary.approved += 1
                elif result.status == IngestionStatus.DUPLICATE:
                    summary.duplicates += 1
                elif result.status == IngestionStatus.QUALITY_FLAGGED:
                    summary.quality_flagged += 1
                elif result.status == IngestionStatus.REJECTED:
                    summary.rejected += 1
                elif result.status == IngestionStatus.ERROR:
                    summary.errors += 1

                # Track embedding source
                if result.has_embedding:
                    if result.embedding_provider == "precomputed":
                        summary.embeddings_precomputed += 1
                    else:
                        summary.embeddings_generated += 1

        # Finalize summary
        summary.total = len(records)
        summary.total_time_ms = int((time.time() - start_time) * 1000)

        # Complete progress tracking
        self._progress_tracker.complete_batch()

        return IngestionBatchResult(
            batch_id=batch_id,
            source_name=source_name,
            records=results,
            summary=summary,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={
                "schema_detected": schema_mapping is not None,
                "max_concurrent": self._max_concurrent,
            },
        )

    async def _process_record(
        self,
        record: AddressRecord,
        schema_mapping: Any | None,
    ) -> IngestionRecord:
        """Process a single address record.

        Args:
            record: AddressRecord to process.
            schema_mapping: Schema mapping to apply.

        Returns:
            Processed IngestionRecord.
        """
        async with self._semaphore:
            start_time = time.time()

            # Create ingestion record
            ing_record = IngestionRecord(
                source_record=record,
                status=IngestionStatus.PENDING,
            )

            try:
                # Step 1: Apply schema mapping
                if schema_mapping and self._schema_mapper:
                    ing_record.parsed_address = self._schema_mapper.apply_mapping(
                        record.raw_data, schema_mapping
                    )
                    ing_record.status = IngestionStatus.SCHEMA_MAPPED

                # Step 2: Get or generate embedding
                if self._embedding_chain:
                    address_text = (
                        ing_record.parsed_address.single_line
                        if ing_record.parsed_address
                        else record.raw_address or ""
                    )

                    result = await self._embedding_chain.embed(
                        text=address_text,
                        precomputed=record.precomputed_embedding,
                    )

                    if result:
                        ing_record.embedding = result.embedding
                        ing_record.embedding_provider = result.provider

                        self._progress_tracker.record_embedding(
                            record.source_id,
                            precomputed=result.provider == "precomputed",
                            provider=result.provider,
                        )

                # Step 3: Run enforcer pipeline
                if self._enforcer_coordinator:
                    enforcer_result = await self._enforcer_coordinator.enforce(
                        ing_record
                    )
                    ing_record.enforcer_results = enforcer_result

                    # Check if duplicate
                    is_duplicate = any(
                        r.get("enforcer_name") == "duplicate_enforcer"
                        and not r.get("passed", True)
                        for r in enforcer_result.get("results", [])
                    )

                    self._progress_tracker.record_processed(
                        record.source_id,
                        approved=enforcer_result.get("approved", False),
                        is_duplicate=is_duplicate,
                    )

                    if is_duplicate:
                        ing_record.status = IngestionStatus.DUPLICATE
                else:
                    # No enforcer - auto approve
                    ing_record.status = IngestionStatus.APPROVED
                    self._progress_tracker.record_processed(
                        record.source_id,
                        approved=True,
                    )

                # Calculate processing time
                ing_record.processing_time_ms = int(
                    (time.time() - start_time) * 1000
                )

            except Exception as e:
                logger.error(f"Error processing record {record.source_id}: {e}")
                ing_record.status = IngestionStatus.ERROR
                ing_record.error_message = str(e)
                ing_record.processing_time_ms = int(
                    (time.time() - start_time) * 1000
                )

            return ing_record

    async def process_stream(
        self,
        adapter: BaseStreamAdapter,
        source_name: str,
        batch_size: int = 1000,
    ) -> AsyncIterator[IngestionBatchResult]:
        """Process records from adapter in batches.

        Args:
            adapter: Data source adapter.
            source_name: Source name for schema caching.
            batch_size: Records per batch.

        Yields:
            IngestionBatchResult for each batch.
        """
        current_batch: list[AddressRecord] = []
        schema_mapping = None

        async for record in adapter.read_stream():
            current_batch.append(record)

            # Detect schema from first batch samples
            if len(current_batch) == min(5, batch_size) and not schema_mapping:
                if self._schema_mapper:
                    sample_records = [r.raw_data for r in current_batch]
                    schema_mapping = await self._schema_mapper.detect_mapping(
                        source_name=source_name,
                        sample_records=sample_records,
                    )
                    logger.info(
                        f"Detected schema for {source_name}: "
                        f"confidence={schema_mapping.confidence:.2f}"
                    )

            # Process batch when full
            if len(current_batch) >= batch_size:
                result = await self.process_batch(
                    current_batch, source_name, schema_mapping
                )
                yield result
                current_batch = []

        # Process remaining records
        if current_batch:
            result = await self.process_batch(
                current_batch, source_name, schema_mapping
            )
            yield result
