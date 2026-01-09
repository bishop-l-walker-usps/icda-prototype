"""Address Data Ingestion System.

This package provides a flexible, adaptive system for ingesting address data
from various sources (NCOA batch files, REST webhooks, file drops) with
support for multiple embedding providers and a 5-stage quality enforcer pipeline.

Key Components:
- Adapters: Connect to various data sources (NCOA, REST, files)
- Embeddings: Multi-provider embedding support with fallback chains
- Schema: AI-powered field mapping for unknown formats
- Enforcers: 5-stage quality validation pipeline
- Pipeline: Main orchestration and batch processing

Example:
    from icda.ingestion import IngestionPipeline, IngestionConfig

    config = IngestionConfig.from_env()
    pipeline = IngestionPipeline(config)
    await pipeline.initialize()

    results = await pipeline.ingest_batch("/path/to/ncoa/data.json")
"""

from icda.ingestion.pipeline.ingestion_models import (
    AddressRecord,
    IngestionRecord,
    IngestionStatus,
    IngestionBatchResult,
    IngestionBatchSummary,
    IngestionEvent,
    IngestionEventData,
)
from icda.ingestion.config.ingestion_config import (
    IngestionConfig,
    IngestionMode,
    AdapterConfig,
    EmbeddingProviderConfig,
    EnforcerConfig,
)
from icda.ingestion.pipeline.ingestion_pipeline import IngestionPipeline
from icda.ingestion.pipeline.batch_processor import BatchProcessor
from icda.ingestion.pipeline.progress_tracker import ProgressTracker
from icda.ingestion.adapters.base_adapter import BaseStreamAdapter
from icda.ingestion.adapters.ncoa_batch_adapter import NCOABatchAdapter
from icda.ingestion.adapters.rest_webhook_adapter import RESTWebhookAdapter
from icda.ingestion.adapters.file_watcher_adapter import FileWatcherAdapter

__all__ = [
    # Models
    "AddressRecord",
    "IngestionRecord",
    "IngestionStatus",
    "IngestionBatchResult",
    "IngestionBatchSummary",
    "IngestionEvent",
    "IngestionEventData",
    # Config
    "IngestionConfig",
    "IngestionMode",
    "AdapterConfig",
    "EmbeddingProviderConfig",
    "EnforcerConfig",
    # Pipeline
    "IngestionPipeline",
    "BatchProcessor",
    "ProgressTracker",
    # Adapters
    "BaseStreamAdapter",
    "NCOABatchAdapter",
    "RESTWebhookAdapter",
    "FileWatcherAdapter",
]
