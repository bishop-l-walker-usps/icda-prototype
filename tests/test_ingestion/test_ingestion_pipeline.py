"""Tests for main ingestion pipeline."""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from icda.ingestion.pipeline.progress_tracker import ProgressTracker
from icda.ingestion.pipeline.ingestion_models import (
    AddressRecord,
    IngestionRecord,
    IngestionStatus,
    IngestionEvent,
    IngestionEventData,
)
from icda.ingestion.config.ingestion_config import (
    IngestionConfig,
    IngestionMode,
    EnforcerConfig,
)


class TestProgressTracker:
    """Test ProgressTracker."""

    @pytest.fixture
    def tracker(self):
        """Create progress tracker."""
        return ProgressTracker(emit_interval=10)

    def test_start_batch(self, tracker):
        """Test starting batch tracking."""
        tracker.start_batch("batch_123", total_records=100)

        assert tracker._batch_id == "batch_123"
        assert tracker._total_records == 100
        assert tracker._processed == 0

    def test_record_processed_approved(self, tracker):
        """Test recording approved record."""
        tracker.start_batch("batch", 10)
        tracker.record_processed("rec_1", approved=True)

        assert tracker._processed == 1
        assert tracker._approved == 1
        assert tracker._rejected == 0

    def test_record_processed_rejected(self, tracker):
        """Test recording rejected record."""
        tracker.start_batch("batch", 10)
        tracker.record_processed("rec_1", approved=False)

        assert tracker._processed == 1
        assert tracker._approved == 0
        assert tracker._rejected == 1

    def test_record_duplicate(self, tracker):
        """Test recording duplicate."""
        tracker.start_batch("batch", 10)
        tracker.record_processed("rec_1", approved=False, is_duplicate=True)

        assert tracker._rejected == 1

    def test_progress_calculation(self, tracker):
        """Test progress percentage."""
        tracker.start_batch("batch", 10)

        for i in range(5):
            tracker.record_processed(f"rec_{i}", approved=True)

        assert tracker.progress == 0.5
        assert tracker._processed == 5

    def test_approval_rate(self, tracker):
        """Test approval rate calculation."""
        tracker.start_batch("batch", 10)

        for i in range(3):
            tracker.record_processed(f"rec_{i}", approved=True)
        for i in range(2):
            tracker.record_processed(f"rej_{i}", approved=False)

        assert tracker.approval_rate == 0.6

    def test_event_listener(self, tracker):
        """Test event listener callback."""
        events = []

        def listener(event_data):
            events.append(event_data)

        tracker.add_listener(listener)
        tracker.start_batch("batch", 10)
        tracker.record_processed("rec_1", approved=True)

        assert len(events) >= 2  # BATCH_STARTED + RECORD_APPROVED

    def test_remove_listener(self, tracker):
        """Test removing event listener."""
        events = []

        def listener(event_data):
            events.append(event_data)

        tracker.add_listener(listener)
        tracker.remove_listener(listener)
        tracker.start_batch("batch", 10)

        assert len(events) == 0

    def test_complete_batch(self, tracker):
        """Test completing batch."""
        tracker.start_batch("batch", 5)
        for i in range(5):
            tracker.record_processed(f"rec_{i}", approved=True)
        tracker.complete_batch()

        stats = tracker.get_stats()
        assert stats["processed"] == 5
        assert stats["progress"] == 1.0


class TestIngestionModels:
    """Test ingestion data models."""

    def test_address_record_creation(self):
        """Test creating AddressRecord."""
        record = AddressRecord(
            source_id="test_1",
            raw_data={"address": "123 Test St"},
            raw_address="123 Test St",
            precomputed_embedding=[0.1] * 1024,
        )

        assert record.source_id == "test_1"
        assert record.raw_address == "123 Test St"
        assert len(record.precomputed_embedding) == 1024

    def test_ingestion_record_status(self):
        """Test IngestionRecord status tracking."""
        source = AddressRecord(source_id="test", raw_data={})
        record = IngestionRecord(source_record=source)

        assert record.status == IngestionStatus.PENDING

        record.status = IngestionStatus.APPROVED
        assert record.status == IngestionStatus.APPROVED

    def test_ingestion_record_has_embedding(self):
        """Test embedding property."""
        source = AddressRecord(source_id="test", raw_data={})
        record = IngestionRecord(source_record=source)

        assert record.has_embedding is False

        record.embedding = [0.1] * 1024
        assert record.has_embedding is True

    def test_ingestion_status_values(self):
        """Test IngestionStatus enum values."""
        assert IngestionStatus.PENDING.value == "pending"
        assert IngestionStatus.APPROVED.value == "approved"
        assert IngestionStatus.REJECTED.value == "rejected"
        assert IngestionStatus.DUPLICATE.value == "duplicate"
        assert IngestionStatus.INDEXED.value == "indexed"

    def test_ingestion_event_values(self):
        """Test IngestionEvent enum values."""
        assert IngestionEvent.BATCH_STARTED.value == "batch_started"
        assert IngestionEvent.BATCH_COMPLETED.value == "batch_completed"
        assert IngestionEvent.RECORD_APPROVED.value == "record_approved"
        assert IngestionEvent.RECORD_REJECTED.value == "record_rejected"


class TestIngestionConfig:
    """Test IngestionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IngestionConfig()

        assert config.mode == IngestionMode.BATCH
        assert config.batch_size == 1000
        assert config.max_concurrent == 10

    def test_config_with_options(self):
        """Test configuration with custom options."""
        config = IngestionConfig(
            mode=IngestionMode.REALTIME,
            batch_size=500,
            max_concurrent=20,
            enable_ai_schema_mapping=True,
        )

        assert config.mode == IngestionMode.REALTIME
        assert config.batch_size == 500
        assert config.max_concurrent == 20
        assert config.enable_ai_schema_mapping is True

    def test_enforcer_config(self):
        """Test enforcer configuration."""
        config = IngestionConfig(
            enforcers=EnforcerConfig(
                enabled=True,
                fail_fast=True,
                similarity_threshold=0.90,
                min_completeness_score=0.7,
            ),
        )

        assert config.enforcers.enabled is True
        assert config.enforcers.fail_fast is True
        assert config.enforcers.similarity_threshold == 0.90
        assert config.enforcers.min_completeness_score == 0.7

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = IngestionConfig(
            batch_size=500,
            max_concurrent=15,
        )

        d = config.to_dict()

        assert d["batch_size"] == 500
        assert d["max_concurrent"] == 15

    def test_ingestion_mode_values(self):
        """Test IngestionMode enum."""
        assert IngestionMode.BATCH.value == "batch"
        assert IngestionMode.REALTIME.value == "realtime"
        assert IngestionMode.HYBRID.value == "hybrid"
