"""Data source adapters for address ingestion.

Adapters connect to various data sources and stream address records
for processing through the ingestion pipeline.

Supported adapters:
- NCOABatchAdapter: C library NCOA output with pre-computed embeddings
- RESTWebhookAdapter: Real-time REST webhook updates
- FileWatcherAdapter: CSV/JSON file drops
"""

from icda.ingestion.adapters.base_adapter import (
    BaseStreamAdapter,
    AdapterType,
    AdapterStats,
    MemoryAdapter,
)
from icda.ingestion.adapters.ncoa_batch_adapter import NCOABatchAdapter
from icda.ingestion.adapters.rest_webhook_adapter import (
    RESTWebhookAdapter,
    WebhookBuffer,
    WebhookRegistry,
)
from icda.ingestion.adapters.file_watcher_adapter import FileWatcherAdapter
from icda.ingestion.pipeline.ingestion_models import AddressRecord

__all__ = [
    # Base
    "BaseStreamAdapter",
    "AdapterType",
    "AdapterStats",
    "MemoryAdapter",
    "AddressRecord",
    # NCOA
    "NCOABatchAdapter",
    # Webhook
    "RESTWebhookAdapter",
    "WebhookBuffer",
    "WebhookRegistry",
    # File watcher
    "FileWatcherAdapter",
]
