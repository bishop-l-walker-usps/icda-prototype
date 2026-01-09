"""Tests for 5-stage ingestion enforcer pipeline."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)
from icda.ingestion.pipeline.ingestion_models import (
    AddressRecord,
    IngestionRecord,
    IngestionStatus,
)


class TestIngestionGate:
    """Test IngestionGate enum."""

    def test_gate_values(self):
        """Test gate enum values."""
        assert IngestionGate.FIELDS_MAPPED.value == "fields_mapped"
        assert IngestionGate.REQUIRED_PRESENT.value == "required_present"
        assert IngestionGate.ADDRESS_PARSEABLE.value == "address_parseable"
        assert IngestionGate.NOT_IN_BATCH.value == "not_in_batch"
        assert IngestionGate.COMPLETENESS_SCORE.value == "completeness_score"
        assert IngestionGate.ALL_GATES_PASSED.value == "all_gates_passed"


class TestIngestionGateResult:
    """Test IngestionGateResult."""

    def test_passed_result(self):
        """Test creating passed gate result."""
        result = IngestionGateResult(
            gate=IngestionGate.FIELDS_MAPPED,
            passed=True,
            message="All fields mapped",
            score=1.0,
        )

        assert result.passed is True
        assert result.gate == IngestionGate.FIELDS_MAPPED
        assert result.score == 1.0

    def test_failed_result(self):
        """Test creating failed gate result."""
        result = IngestionGateResult(
            gate=IngestionGate.REQUIRED_PRESENT,
            passed=False,
            message="Missing street field",
            details={"missing_fields": ["street"]},
        )

        assert result.passed is False
        assert "street" in result.details["missing_fields"]


class TestIngestionModels:
    """Test basic ingestion models."""

    def test_address_record(self):
        """Test AddressRecord creation."""
        record = AddressRecord(
            source_id="test_1",
            raw_data={"street": "123 Main St"},
            raw_address="123 Main St, Springfield, IL 62701",
        )

        assert record.source_id == "test_1"
        assert record.raw_address == "123 Main St, Springfield, IL 62701"

    def test_ingestion_record(self):
        """Test IngestionRecord creation."""
        source = AddressRecord(source_id="test", raw_data={})
        record = IngestionRecord(source_record=source)

        assert record.status == IngestionStatus.PENDING
        assert record.source_id == "test"

    def test_ingestion_status(self):
        """Test IngestionStatus values."""
        assert IngestionStatus.PENDING.value == "pending"
        assert IngestionStatus.APPROVED.value == "approved"
        assert IngestionStatus.REJECTED.value == "rejected"
