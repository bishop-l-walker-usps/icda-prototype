"""REST webhook adapter for real-time address data ingestion.

Receives address data via HTTP webhooks and streams to pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator
from collections import deque

from icda.ingestion.adapters.base_adapter import BaseStreamAdapter, AdapterType
from icda.ingestion.pipeline.ingestion_models import AddressRecord

logger = logging.getLogger(__name__)


class WebhookBuffer:
    """Thread-safe buffer for webhook payloads.

    Features:
    - Async queue for producer/consumer pattern
    - Configurable max size with backpressure
    - Timeout support for graceful shutdown
    """

    __slots__ = (
        "_queue",
        "_max_size",
        "_closed",
    )

    def __init__(self, max_size: int = 10000):
        """Initialize buffer.

        Args:
            max_size: Maximum buffer size before backpressure.
        """
        self._queue: asyncio.Queue[AddressRecord | None] = asyncio.Queue(
            maxsize=max_size
        )
        self._max_size = max_size
        self._closed = False

    async def put(self, record: AddressRecord) -> bool:
        """Add record to buffer.

        Args:
            record: AddressRecord to buffer.

        Returns:
            True if added, False if buffer closed.
        """
        if self._closed:
            return False

        try:
            await asyncio.wait_for(
                self._queue.put(record),
                timeout=5.0,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Buffer full - dropping record")
            return False

    async def get(self, timeout: float = 1.0) -> AddressRecord | None:
        """Get record from buffer.

        Args:
            timeout: Max wait time in seconds.

        Returns:
            AddressRecord or None if timeout/closed.
        """
        try:
            return await asyncio.wait_for(
                self._queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Close buffer for new records."""
        self._closed = True
        # Signal consumers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self._queue.qsize()

    @property
    def is_closed(self) -> bool:
        """Check if buffer is closed."""
        return self._closed


class RESTWebhookAdapter(BaseStreamAdapter):
    """Adapter for receiving address data via REST webhooks.

    This adapter provides a buffer that FastAPI endpoints can push to,
    and the ingestion pipeline can consume from.

    Usage:
        adapter = RESTWebhookAdapter()
        await adapter.connect()

        # In FastAPI endpoint:
        @app.post("/webhook/address")
        async def receive_address(payload: dict):
            await adapter.receive_payload(payload)

        # In pipeline:
        async for record in adapter.read_stream():
            process(record)
    """

    __slots__ = (
        "_buffer",
        "_max_buffer_size",
        "_connected",
        "_source_name",
        "_records_received",
        "_records_streamed",
        "_last_receive_time",
        "_schema_hint",
    )

    def __init__(
        self,
        source_name: str = "webhook",
        max_buffer_size: int = 10000,
        schema_hint: str | None = None,
    ):
        """Initialize webhook adapter.

        Args:
            source_name: Name for this webhook source.
            max_buffer_size: Max records to buffer.
            schema_hint: Optional hint for schema detection.
        """
        super().__init__(name=source_name)
        self._buffer: WebhookBuffer | None = None
        self._max_buffer_size = max_buffer_size
        self._source_name = source_name
        self._records_received = 0
        self._records_streamed = 0
        self._last_receive_time: float | None = None
        self._schema_hint = schema_hint

    @property
    def adapter_type(self) -> AdapterType:
        """Return adapter type identifier."""
        return AdapterType.REST_WEBHOOK

    def has_precomputed_embeddings(self) -> bool:
        """Check if source provides pre-computed embeddings."""
        return False  # Webhooks may or may not have embeddings

    @property
    def source_name(self) -> str:
        """Get source identifier."""
        return self._source_name

    @property
    def source_type(self) -> str:
        """Get source type."""
        return "rest_webhook"

    async def connect(self) -> bool:
        """Initialize webhook buffer.

        Returns:
            True if connected.
        """
        self._buffer = WebhookBuffer(max_size=self._max_buffer_size)
        self._connected = True
        logger.info(f"Webhook adapter '{self._source_name}' connected")
        return True

    async def disconnect(self) -> None:
        """Close webhook buffer."""
        if self._buffer:
            self._buffer.close()
        self._connected = False
        logger.info(
            f"Webhook adapter '{self._source_name}' disconnected: "
            f"{self._records_received} received, {self._records_streamed} streamed"
        )

    async def read_stream(self) -> AsyncIterator[AddressRecord]:
        """Stream records from webhook buffer.

        Yields:
            AddressRecord instances.
        """
        if not self._buffer:
            return

        while self._connected or self._buffer.size > 0:
            record = await self._buffer.get(timeout=1.0)

            if record is None:
                if self._buffer.is_closed and self._buffer.size == 0:
                    break
                continue

            self._records_streamed += 1
            yield record

    async def read_batch(
        self,
        batch_size: int = 1000,
        timeout: float = 5.0,
    ) -> list[AddressRecord]:
        """Read a batch of records from buffer.

        Args:
            batch_size: Max records to return.
            timeout: Max wait time in seconds.

        Returns:
            List of AddressRecords.
        """
        if not self._buffer:
            return []

        records: list[AddressRecord] = []
        start_time = time.time()

        while len(records) < batch_size:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                break

            record = await self._buffer.get(timeout=min(remaining, 0.5))

            if record is None:
                if self._buffer.is_closed:
                    break
                continue

            records.append(record)
            self._records_streamed += 1

        return records

    async def receive_payload(
        self,
        payload: dict[str, Any],
        source_id: str | None = None,
    ) -> bool:
        """Receive a webhook payload.

        Args:
            payload: Raw address data payload.
            source_id: Optional external ID.

        Returns:
            True if accepted, False if buffer full/closed.
        """
        if not self._buffer or not self._connected:
            return False

        # Create AddressRecord from payload
        record = self._payload_to_record(payload, source_id)

        success = await self._buffer.put(record)

        if success:
            self._records_received += 1
            self._last_receive_time = time.time()

        return success

    async def receive_batch(
        self,
        payloads: list[dict[str, Any]],
    ) -> int:
        """Receive multiple payloads.

        Args:
            payloads: List of address data payloads.

        Returns:
            Number of payloads accepted.
        """
        accepted = 0

        for i, payload in enumerate(payloads):
            source_id = payload.get("id") or f"batch_{i}"
            if await self.receive_payload(payload, source_id):
                accepted += 1

        return accepted

    def _payload_to_record(
        self,
        payload: dict[str, Any],
        source_id: str | None,
    ) -> AddressRecord:
        """Convert webhook payload to AddressRecord.

        Args:
            payload: Raw payload data.
            source_id: Optional external ID.

        Returns:
            AddressRecord instance.
        """
        # Extract ID from various common fields
        record_id = source_id or payload.get("id") or payload.get(
            "record_id"
        ) or payload.get("address_id") or f"webhook_{self._records_received}"

        # Try to extract raw address string
        raw_address = None
        for key in ["address", "full_address", "address_line", "street_address"]:
            if key in payload:
                raw_address = str(payload[key])
                break

        # Check for pre-computed embedding
        embedding = None
        if "embedding" in payload:
            emb = payload["embedding"]
            if isinstance(emb, list) and len(emb) > 0:
                embedding = [float(x) for x in emb]

        return AddressRecord(
            source_id=str(record_id),
            raw_data=payload,
            raw_address=raw_address,
            precomputed_embedding=embedding,
            source_metadata={
                "source": self._source_name,
                "type": "webhook",
                "received_at": time.time(),
                "schema_hint": self._schema_hint,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "source_name": self._source_name,
            "connected": self._connected,
            "buffer_size": self._buffer.size if self._buffer else 0,
            "max_buffer_size": self._max_buffer_size,
            "records_received": self._records_received,
            "records_streamed": self._records_streamed,
            "pending": self._records_received - self._records_streamed,
            "last_receive_time": self._last_receive_time,
        }

    def get_info(self) -> dict[str, Any]:
        """Get adapter information."""
        return {
            "source_name": self._source_name,
            "source_type": self.source_type,
            "connected": self._connected,
            "schema_hint": self._schema_hint,
            "stats": self.get_stats(),
        }


class WebhookRegistry:
    """Registry for managing multiple webhook adapters.

    Allows routing different webhook sources to different adapters.
    """

    __slots__ = ("_adapters",)

    def __init__(self):
        """Initialize registry."""
        self._adapters: dict[str, RESTWebhookAdapter] = {}

    def register(
        self,
        source_name: str,
        adapter: RESTWebhookAdapter,
    ) -> None:
        """Register an adapter.

        Args:
            source_name: Source identifier.
            adapter: Webhook adapter instance.
        """
        self._adapters[source_name] = adapter
        logger.info(f"Registered webhook adapter: {source_name}")

    def unregister(self, source_name: str) -> None:
        """Unregister an adapter.

        Args:
            source_name: Source identifier.
        """
        if source_name in self._adapters:
            del self._adapters[source_name]
            logger.info(f"Unregistered webhook adapter: {source_name}")

    def get(self, source_name: str) -> RESTWebhookAdapter | None:
        """Get adapter by name.

        Args:
            source_name: Source identifier.

        Returns:
            Adapter if found.
        """
        return self._adapters.get(source_name)

    async def route_payload(
        self,
        source_name: str,
        payload: dict[str, Any],
    ) -> bool:
        """Route payload to appropriate adapter.

        Args:
            source_name: Target source.
            payload: Address data.

        Returns:
            True if routed successfully.
        """
        adapter = self._adapters.get(source_name)
        if not adapter:
            logger.warning(f"No adapter registered for source: {source_name}")
            return False

        return await adapter.receive_payload(payload)

    def list_sources(self) -> list[str]:
        """List registered source names."""
        return list(self._adapters.keys())

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all adapters."""
        return {
            name: adapter.get_stats()
            for name, adapter in self._adapters.items()
        }
