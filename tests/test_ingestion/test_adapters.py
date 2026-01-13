"""Tests for data source adapters."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from icda.ingestion.adapters.base_adapter import (
    BaseStreamAdapter,
    MemoryAdapter,
    AdapterType,
)
from icda.ingestion.adapters.ncoa_batch_adapter import NCOABatchAdapter
from icda.ingestion.adapters.rest_webhook_adapter import (
    RESTWebhookAdapter,
    WebhookBuffer,
    WebhookRegistry,
)
from icda.ingestion.adapters.file_watcher_adapter import FileWatcherAdapter
from icda.ingestion.pipeline.ingestion_models import AddressRecord


class TestMemoryAdapter:
    """Test MemoryAdapter for testing purposes."""

    @pytest.fixture
    def sample_records(self):
        """Create sample address records."""
        return [
            AddressRecord(
                source_id=f"rec_{i}",
                raw_data={"address": f"{i} Main St"},
                raw_address=f"{i} Main St",
            )
            for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, sample_records):
        """Test adapter connection lifecycle."""
        adapter = MemoryAdapter(records=sample_records)

        assert await adapter.connect() is True
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_read_stream(self, sample_records):
        """Test streaming records."""
        adapter = MemoryAdapter(records=sample_records)
        await adapter.connect()

        records = []
        async for record in adapter.read_stream():
            records.append(record)

        assert len(records) == 5
        assert records[0].source_id == "rec_0"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_read_batch(self, sample_records):
        """Test batch reading."""
        adapter = MemoryAdapter(records=sample_records)
        await adapter.connect()

        batch = await adapter.read_batch(batch_size=3)
        assert len(batch) == 3

        await adapter.disconnect()


class TestNCOABatchAdapter:
    """Test NCOABatchAdapter."""

    @pytest.fixture
    def json_data_file(self, tmp_path):
        """Create temp JSON data file."""
        data = [
            {
                "id": "ncoa_1",
                "old_address": "123 Old St",
                "new_address": "456 New Ave",
                "city": "Springfield",
                "state": "IL",
                "zip": "62701",
            },
            {
                "id": "ncoa_2",
                "old_address": "789 Former Rd",
                "new_address": "321 Current Blvd",
                "city": "Chicago",
                "state": "IL",
                "zip": "60601",
            },
        ]

        file_path = tmp_path / "ncoa_data.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        return str(file_path)

    @pytest.fixture
    def csv_data_file(self, tmp_path):
        """Create temp CSV data file."""
        csv_content = """id,old_address,new_address,city,state,zip
ncoa_1,123 Old St,456 New Ave,Springfield,IL,62701
ncoa_2,789 Former Rd,321 Current Blvd,Chicago,IL,60601"""

        file_path = tmp_path / "ncoa_data.csv"
        with open(file_path, "w") as f:
            f.write(csv_content)

        return str(file_path)

    @pytest.mark.asyncio
    async def test_read_json_file(self, json_data_file):
        """Test reading JSON format file."""
        adapter = NCOABatchAdapter(
            input_path=json_data_file,
            file_format="json",
        )

        assert await adapter.connect() is True

        records = await adapter.read_batch(batch_size=100)
        assert len(records) == 2
        assert records[0].source_id == "ncoa_1"
        assert "old_address" in records[0].raw_data

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_read_csv_file(self, csv_data_file):
        """Test reading CSV format file."""
        adapter = NCOABatchAdapter(
            input_path=csv_data_file,
            file_format="csv",
        )

        assert await adapter.connect() is True

        records = await adapter.read_batch(batch_size=100)
        assert len(records) == 2
        assert records[0].raw_data["city"] == "Springfield"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_stream_records(self, json_data_file):
        """Test streaming records from file."""
        adapter = NCOABatchAdapter(
            input_path=json_data_file,
            file_format="json",
        )

        await adapter.connect()

        records = []
        async for record in adapter.read_stream():
            records.append(record)

        assert len(records) == 2
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path):
        """Test handling of missing file."""
        adapter = NCOABatchAdapter(
            input_path=str(tmp_path / "nonexistent.json"),
        )

        result = await adapter.connect()
        assert result is False


class TestRESTWebhookAdapter:
    """Test RESTWebhookAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create webhook adapter."""
        return RESTWebhookAdapter(
            source_name="test_webhook",
            max_buffer_size=100,
        )

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, adapter):
        """Test adapter lifecycle."""
        assert await adapter.connect() is True
        assert adapter._connected is True

        await adapter.disconnect()
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_receive_payload(self, adapter):
        """Test receiving webhook payload."""
        await adapter.connect()

        payload = {
            "id": "webhook_1",
            "address": "123 Test St",
            "city": "Testville",
        }

        result = await adapter.receive_payload(payload)
        assert result is True

        stats = adapter.get_stats()
        assert stats["records_received"] == 1

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_receive_with_embedding(self, adapter):
        """Test receiving payload with embedding."""
        await adapter.connect()

        payload = {
            "id": "webhook_2",
            "address": "456 Vector Ave",
            "embedding": [0.1] * 1024,
        }

        await adapter.receive_payload(payload)

        # Read the record
        batch = await adapter.read_batch(batch_size=1, timeout=1.0)
        assert len(batch) == 1
        assert batch[0].precomputed_embedding is not None
        assert len(batch[0].precomputed_embedding) == 1024

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_batch_receive(self, adapter):
        """Test receiving multiple payloads."""
        await adapter.connect()

        payloads = [
            {"id": f"batch_{i}", "address": f"{i} Batch St"}
            for i in range(5)
        ]

        accepted = await adapter.receive_batch(payloads)
        assert accepted == 5

        stats = adapter.get_stats()
        assert stats["records_received"] == 5

        await adapter.disconnect()


class TestWebhookBuffer:
    """Test WebhookBuffer."""

    @pytest.mark.asyncio
    async def test_put_get(self):
        """Test basic put and get operations."""
        buffer = WebhookBuffer(max_size=10)

        record = AddressRecord(
            source_id="test",
            raw_data={"test": "data"},
        )

        result = await buffer.put(record)
        assert result is True
        assert buffer.size == 1

        retrieved = await buffer.get(timeout=1.0)
        assert retrieved is not None
        assert retrieved.source_id == "test"
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_close_buffer(self):
        """Test buffer close behavior."""
        buffer = WebhookBuffer(max_size=10)

        record = AddressRecord(source_id="test", raw_data={})
        await buffer.put(record)

        buffer.close()
        assert buffer.is_closed is True

        # Should not accept new records
        result = await buffer.put(record)
        assert result is False


class TestWebhookRegistry:
    """Test WebhookRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_route(self):
        """Test registering adapters and routing payloads."""
        registry = WebhookRegistry()

        adapter1 = RESTWebhookAdapter(source_name="source1")
        adapter2 = RESTWebhookAdapter(source_name="source2")

        await adapter1.connect()
        await adapter2.connect()

        registry.register("source1", adapter1)
        registry.register("source2", adapter2)

        assert registry.list_sources() == ["source1", "source2"]

        # Route payload
        result = await registry.route_payload("source1", {"id": "1"})
        assert result is True

        # Invalid source
        result = await registry.route_payload("invalid", {"id": "2"})
        assert result is False

        await adapter1.disconnect()
        await adapter2.disconnect()


class TestFileWatcherAdapter:
    """Test FileWatcherAdapter."""

    @pytest.fixture
    def watch_dir(self, tmp_path):
        """Create temp watch directory."""
        watch = tmp_path / "watch"
        watch.mkdir()
        return watch

    @pytest.fixture
    def archive_dir(self, tmp_path):
        """Create temp archive directory."""
        archive = tmp_path / "archive"
        archive.mkdir()
        return archive

    @pytest.mark.asyncio
    async def test_connect_creates_dirs(self, tmp_path):
        """Test that connect creates watch directories."""
        watch_dir = tmp_path / "new_watch"

        adapter = FileWatcherAdapter(
            watch_dirs=[str(watch_dir)],
            poll_interval=0.1,
        )

        await adapter.connect()
        assert watch_dir.exists()

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_process_json_file(self, watch_dir, archive_dir):
        """Test processing dropped JSON file."""
        adapter = FileWatcherAdapter(
            watch_dirs=[str(watch_dir)],
            archive_dir=str(archive_dir),
            poll_interval=0.1,
        )

        await adapter.connect()

        # Drop a JSON file
        data = [
            {"id": "1", "address": "123 Test St"},
            {"id": "2", "address": "456 Demo Ave"},
        ]
        json_file = watch_dir / "test.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Read records
        records = await adapter.read_batch(batch_size=10)
        assert len(records) == 2

        # File should be archived
        assert not json_file.exists()
        archived_files = list(archive_dir.glob("test_*.json"))
        assert len(archived_files) == 1

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_process_csv_file(self, watch_dir):
        """Test processing dropped CSV file."""
        adapter = FileWatcherAdapter(
            watch_dirs=[str(watch_dir)],
            poll_interval=0.1,
        )

        await adapter.connect()

        # Drop a CSV file
        csv_content = """id,address,city
1,123 Test St,Springfield
2,456 Demo Ave,Chicago"""

        csv_file = watch_dir / "addresses.csv"
        with open(csv_file, "w") as f:
            f.write(csv_content)

        # Wait for processing
        await asyncio.sleep(0.3)

        records = await adapter.read_batch(batch_size=10)
        assert len(records) == 2
        assert records[0].raw_data["city"] == "Springfield"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_manual_add_file(self, watch_dir):
        """Test manually adding a file to process."""
        adapter = FileWatcherAdapter(
            watch_dirs=[str(watch_dir)],
            poll_interval=10.0,  # Long interval
        )

        await adapter.connect()

        # Create file outside watch dir
        data = [{"id": "manual", "address": "Manual St"}]
        json_file = watch_dir / "manual.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        # Manually add
        result = adapter.add_file(json_file)
        assert result is True

        # Adding again should fail
        result = adapter.add_file(json_file)
        assert result is False

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_get_stats(self, watch_dir):
        """Test statistics tracking."""
        adapter = FileWatcherAdapter(
            watch_dirs=[str(watch_dir)],
            poll_interval=0.1,
        )

        await adapter.connect()

        stats = adapter.get_stats()
        assert stats["files_discovered"] == 0
        assert stats["connected"] is True

        await adapter.disconnect()


class TestAdapterType:
    """Test AdapterType enum."""

    def test_adapter_types(self):
        """Test adapter type values."""
        assert AdapterType.NCOA_BATCH.value == "ncoa_batch"
        assert AdapterType.REST_WEBHOOK.value == "rest_webhook"
        assert AdapterType.FILE_WATCHER.value == "file_watcher"
        assert AdapterType.MEMORY.value == "memory"
