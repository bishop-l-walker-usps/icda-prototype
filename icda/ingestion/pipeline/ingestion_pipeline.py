"""Main ingestion pipeline.

Orchestrates the complete address data ingestion flow:
Adapter -> Schema Mapper -> Embedding Chain -> Enforcer Pipeline -> Index
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, TYPE_CHECKING

from icda.ingestion.config.ingestion_config import (
    IngestionConfig,
    IngestionMode,
)
from icda.ingestion.pipeline.ingestion_models import (
    IngestionBatchResult,
    IngestionRecord,
    IngestionStatus,
)
from icda.ingestion.pipeline.progress_tracker import ProgressTracker
from icda.ingestion.pipeline.batch_processor import BatchProcessor
from icda.ingestion.adapters.ncoa_batch_adapter import NCOABatchAdapter
from icda.ingestion.embeddings.provider_chain import EmbeddingProviderChain
from icda.ingestion.embeddings.precomputed_provider import PrecomputedEmbeddingProvider
from icda.ingestion.embeddings.titan_provider import TitanEmbeddingProvider
from icda.ingestion.schema.schema_mapper import AISchemaMapper
from icda.ingestion.schema.mapping_cache import MappingCache
from icda.ingestion.enforcers.ingestion_coordinator import IngestionEnforcerCoordinator

if TYPE_CHECKING:
    from icda.address_normalizer import AddressNormalizer
    from icda.address_index import AddressIndex
    from icda.indexes.address_vector_index import AddressVectorIndex
    from icda.embeddings import EmbeddingClient
    from icda.nova import NovaClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main ingestion pipeline orchestrator.

    Coordinates all ingestion components:
    - Data source adapters
    - AI schema mapping
    - Embedding provider chain
    - 5-stage enforcer pipeline
    - Index integration

    Usage:
        config = IngestionConfig.from_env()
        pipeline = IngestionPipeline(config)
        await pipeline.initialize()

        # Process NCOA batch file
        result = await pipeline.ingest_file("/path/to/ncoa/data.json")

        # Or stream processing
        async for batch_result in pipeline.ingest_stream(adapter):
            print(f"Processed batch: {batch_result.summary.to_dict()}")
    """

    __slots__ = (
        "_config",
        "_initialized",
        "_progress_tracker",
        "_schema_mapper",
        "_mapping_cache",
        "_embedding_chain",
        "_enforcer_coordinator",
        "_batch_processor",
        "_address_index",
        "_vector_index",
        "_stats",
    )

    def __init__(
        self,
        config: IngestionConfig | None = None,
        address_normalizer: AddressNormalizer | None = None,
        address_index: AddressIndex | None = None,
        vector_index: AddressVectorIndex | None = None,
        embedding_client: EmbeddingClient | None = None,
        nova_client: NovaClient | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            config: Pipeline configuration.
            address_normalizer: For address parsing in enforcers.
            address_index: For duplicate detection and indexing.
            vector_index: For semantic search indexing.
            embedding_client: Existing Titan embedding client.
            nova_client: For AI schema detection.
        """
        self._config = config or IngestionConfig()
        self._initialized = False

        # Will be initialized in initialize()
        self._progress_tracker: ProgressTracker | None = None
        self._schema_mapper: AISchemaMapper | None = None
        self._mapping_cache: MappingCache | None = None
        self._embedding_chain: EmbeddingProviderChain | None = None
        self._enforcer_coordinator: IngestionEnforcerCoordinator | None = None
        self._batch_processor: BatchProcessor | None = None

        # External dependencies
        self._address_index = address_index
        self._vector_index = vector_index

        # Stats
        self._stats = {
            "batches_processed": 0,
            "records_processed": 0,
            "records_approved": 0,
            "records_rejected": 0,
            "records_indexed": 0,
        }

        # Store for initialization
        self._init_params = {
            "address_normalizer": address_normalizer,
            "embedding_client": embedding_client,
            "nova_client": nova_client,
        }

    async def initialize(self) -> bool:
        """Initialize all pipeline components.

        Returns:
            True if initialization successful.
        """
        logger.info("Initializing ingestion pipeline...")

        # 1. Progress tracker
        self._progress_tracker = ProgressTracker(
            emit_interval=self._config.progress_interval
        )

        # 2. Schema mapping cache
        self._mapping_cache = MappingCache(
            cache_path=self._config.schema_cache_path
        )
        await self._mapping_cache.initialize()

        # 3. AI Schema mapper
        if self._config.enable_ai_schema_mapping:
            self._schema_mapper = AISchemaMapper(
                nova_client=self._init_params.get("nova_client"),
                cache=self._mapping_cache,
            )

        # 4. Embedding provider chain
        providers = await self._build_embedding_providers()
        if providers:
            self._embedding_chain = EmbeddingProviderChain(
                providers=providers,
                target_dimension=self._config.embeddings.target_dimension,
                enable_normalization=self._config.embeddings.enable_normalization,
                circuit_threshold=self._config.embeddings.circuit_breaker_threshold,
                circuit_timeout=self._config.embeddings.circuit_breaker_timeout,
            )
            await self._embedding_chain.initialize()

        # 5. Enforcer coordinator
        if self._config.enforcers.enabled:
            self._enforcer_coordinator = IngestionEnforcerCoordinator(
                address_normalizer=self._init_params.get("address_normalizer"),
                address_index=self._address_index,
                enabled=True,
                fail_fast=self._config.enforcers.fail_fast,
                required_fields=self._config.enforcers.required_fields,
                similarity_threshold=self._config.enforcers.similarity_threshold,
                check_existing_index=self._config.enforcers.check_existing_index,
                min_completeness=self._config.enforcers.min_completeness_score,
                min_confidence=self._config.enforcers.min_confidence_threshold,
                require_pr_urbanization=self._config.enforcers.require_pr_urbanization,
                require_embedding=self._config.enforcers.require_embedding,
                min_quality_score=self._config.enforcers.min_quality_score,
            )

        # 6. Batch processor
        self._batch_processor = BatchProcessor(
            schema_mapper=self._schema_mapper,
            embedding_chain=self._embedding_chain,
            enforcer_coordinator=self._enforcer_coordinator,
            progress_tracker=self._progress_tracker,
            max_concurrent=self._config.max_concurrent,
        )

        self._initialized = True
        logger.info("Ingestion pipeline initialized successfully")

        return True

    async def _build_embedding_providers(self) -> list:
        """Build embedding provider list based on config."""
        from icda.ingestion.embeddings.base_provider import BaseEmbeddingProvider

        providers: list[BaseEmbeddingProvider] = []
        emb_config = self._config.embeddings

        # Add precomputed provider first (for NCOA with C library embeddings)
        precomputed = PrecomputedEmbeddingProvider(
            expected_dimension=emb_config.target_dimension,
        )
        await precomputed.initialize()
        providers.append(precomputed)

        # Add Titan if existing client available
        existing_client = self._init_params.get("embedding_client")
        if existing_client:
            titan = TitanEmbeddingProvider(
                existing_client=existing_client,
                region=emb_config.titan_region,
                model=emb_config.titan_model,
            )
            await titan.initialize()
            if titan.available:
                providers.append(titan)

        # Add other providers based on fallback order
        for provider_name in emb_config.fallback_order:
            if provider_name == "titan" and existing_client:
                continue  # Already added

            try:
                provider = await self._create_provider(provider_name)
                if provider and provider.available:
                    providers.append(provider)
            except Exception as e:
                logger.warning(f"Failed to create {provider_name} provider: {e}")

        logger.info(
            f"Embedding providers initialized: "
            f"{[p.provider_name for p in providers]}"
        )

        return providers

    async def _create_provider(self, provider_name: str):
        """Create an embedding provider by name."""
        emb_config = self._config.embeddings

        if provider_name == "openai" and emb_config.openai_api_key:
            from icda.ingestion.embeddings.openai_provider import OpenAIEmbeddingProvider
            provider = OpenAIEmbeddingProvider(
                api_key=emb_config.openai_api_key,
                model=emb_config.openai_model,
            )
            await provider.initialize()
            return provider

        elif provider_name == "cohere" and emb_config.cohere_api_key:
            from icda.ingestion.embeddings.cohere_provider import CohereEmbeddingProvider
            provider = CohereEmbeddingProvider(
                api_key=emb_config.cohere_api_key,
                model=emb_config.cohere_model,
            )
            await provider.initialize()
            return provider

        elif provider_name == "voyage" and emb_config.voyage_api_key:
            from icda.ingestion.embeddings.voyage_provider import VoyageEmbeddingProvider
            provider = VoyageEmbeddingProvider(
                api_key=emb_config.voyage_api_key,
                model=emb_config.voyage_model,
            )
            await provider.initialize()
            return provider

        elif provider_name == "sentence_transformers":
            from icda.ingestion.embeddings.sentence_transformer import (
                SentenceTransformerProvider
            )
            provider = SentenceTransformerProvider(
                model_name=emb_config.sentence_transformer_model,
                device=emb_config.sentence_transformer_device,
            )
            await provider.initialize()
            return provider

        return None

    async def ingest_file(
        self,
        file_path: str,
        embedding_path: str | None = None,
        file_format: str = "json",
    ) -> IngestionBatchResult:
        """Ingest address data from a file.

        Args:
            file_path: Path to address data file.
            embedding_path: Optional path to pre-computed embeddings.
            file_format: File format (json, csv).

        Returns:
            IngestionBatchResult with all processed records.
        """
        if not self._initialized:
            await self.initialize()

        # Create NCOA adapter
        adapter = NCOABatchAdapter(
            input_path=file_path,
            embedding_path=embedding_path,
            embedding_dim=self._config.embeddings.target_dimension,
            file_format=file_format,
        )

        await adapter.connect()

        try:
            # Read all records
            records = await adapter.read_batch(batch_size=100000)  # Large batch

            # Process batch
            result = await self._batch_processor.process_batch(
                records=records,
                source_name=file_path,
            )

            # Index approved records
            if self._config.update_address_index or self._config.update_vector_index:
                await self._index_approved_records(result.approved_records)

            # Update stats
            self._stats["batches_processed"] += 1
            self._stats["records_processed"] += result.summary.total
            self._stats["records_approved"] += result.summary.approved
            self._stats["records_rejected"] += result.summary.rejected

            return result

        finally:
            await adapter.disconnect()

    async def ingest_stream(
        self,
        adapter,
        source_name: str,
    ) -> AsyncIterator[IngestionBatchResult]:
        """Process records from adapter in streaming batches.

        Args:
            adapter: Data source adapter.
            source_name: Name for schema caching.

        Yields:
            IngestionBatchResult for each batch.
        """
        if not self._initialized:
            await self.initialize()

        async for batch_result in self._batch_processor.process_stream(
            adapter=adapter,
            source_name=source_name,
            batch_size=self._config.batch_size,
        ):
            # Index approved records
            if self._config.update_address_index or self._config.update_vector_index:
                await self._index_approved_records(batch_result.approved_records)

            # Update stats
            self._stats["batches_processed"] += 1
            self._stats["records_processed"] += batch_result.summary.total
            self._stats["records_approved"] += batch_result.summary.approved
            self._stats["records_rejected"] += batch_result.summary.rejected

            yield batch_result

    async def _index_approved_records(
        self,
        records: list[IngestionRecord],
    ) -> int:
        """Index approved records into address and vector indexes.

        Args:
            records: List of approved IngestionRecords.

        Returns:
            Number of records indexed.
        """
        indexed_count = 0

        for record in records:
            if not record.parsed_address:
                continue

            try:
                # Add to address index
                if self._config.update_address_index and self._address_index:
                    self._address_index._add_address(
                        record.parsed_address,
                        record.customer_id or record.source_id,
                        source="ingestion",
                    )

                # Add to vector index
                if (
                    self._config.update_vector_index
                    and self._vector_index
                    and record.has_embedding
                ):
                    await self._vector_index.index_address_with_embedding(
                        parsed=record.parsed_address,
                        customer_id=record.customer_id or record.source_id,
                        embedding=record.embedding,
                    )

                record.status = IngestionStatus.INDEXED
                indexed_count += 1

                if self._progress_tracker:
                    self._progress_tracker.record_indexed(
                        record.source_id,
                        index_type="address+vector" if record.has_embedding else "address",
                    )

            except Exception as e:
                logger.error(f"Failed to index record {record.source_id}: {e}")

        self._stats["records_indexed"] += indexed_count
        return indexed_count

    def add_progress_listener(self, callback) -> None:
        """Add progress event listener.

        Args:
            callback: Function to call with event data.
        """
        if self._progress_tracker:
            self._progress_tracker.add_listener(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = self._stats.copy()

        if self._embedding_chain:
            stats["embedding_chain"] = self._embedding_chain.get_info()

        if self._enforcer_coordinator:
            stats["enforcer"] = self._enforcer_coordinator.get_info()

        return stats

    def get_info(self) -> dict[str, Any]:
        """Get pipeline information."""
        return {
            "initialized": self._initialized,
            "config": self._config.to_dict(),
            "stats": self._stats,
            "embedding_providers": (
                self._embedding_chain.available_providers
                if self._embedding_chain
                else []
            ),
            "schema_cache_entries": (
                len(self._mapping_cache.list_sources())
                if self._mapping_cache
                else 0
            ),
        }
