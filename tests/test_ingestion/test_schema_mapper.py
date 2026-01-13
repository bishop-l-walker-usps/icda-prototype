"""Tests for AI schema mapper."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from icda.ingestion.schema.schema_models import (
    FieldMapping,
    SchemaMapping,
    MappingConfidence,
    CompositeFieldRule,
)
from icda.ingestion.schema.mapping_cache import MappingCache


class TestFieldMapping:
    """Test FieldMapping dataclass."""

    def test_simple_mapping(self):
        """Test creating simple field mapping."""
        mapping = FieldMapping(
            source_field="addr",
            target_field="street",
            confidence=0.95,
        )

        assert mapping.source_field == "addr"
        assert mapping.target_field == "street"
        assert mapping.confidence == 0.95

    def test_mapping_with_transform(self):
        """Test mapping with transform function."""
        mapping = FieldMapping(
            source_field="state_abbr",
            target_field="state",
            confidence=0.9,
            transform="uppercase",
        )

        assert mapping.transform == "uppercase"

    def test_to_dict(self):
        """Test converting mapping to dict."""
        mapping = FieldMapping(
            source_field="zip",
            target_field="postal_code",
            confidence=0.85,
        )

        d = mapping.to_dict()
        assert d["source_field"] == "zip"
        assert d["target_field"] == "postal_code"
        assert d["confidence"] == 0.85


class TestSchemaMapping:
    """Test SchemaMapping dataclass."""

    @pytest.fixture
    def sample_mapping(self):
        """Create sample schema mapping."""
        return SchemaMapping(
            source_name="ncoa_feed",
            source_format="json",
            field_mappings=[
                FieldMapping("street", "street_name", 0.95),
                FieldMapping("city", "city", 0.99),
                FieldMapping("st", "state", 0.85),
                FieldMapping("zipcode", "zip_code", 0.90),
            ],
            confidence=0.92,
        )

    def test_mapping_properties(self, sample_mapping):
        """Test schema mapping properties."""
        assert sample_mapping.source_name == "ncoa_feed"
        assert len(sample_mapping.field_mappings) == 4
        assert sample_mapping.confidence == 0.92

    def test_get_mapping_for_target(self, sample_mapping):
        """Test getting mapping for target field."""
        mapping = sample_mapping.get_mapping("state")

        assert mapping is not None
        assert mapping.source_field == "st"
        assert mapping.confidence == 0.85

    def test_to_dict(self, sample_mapping):
        """Test converting to dictionary."""
        d = sample_mapping.to_dict()

        assert d["source_name"] == "ncoa_feed"
        assert d["confidence"] == 0.92
        assert len(d["field_mappings"]) == 4


class TestCompositeFieldRule:
    """Test CompositeFieldRule."""

    def test_concatenate_rule(self):
        """Test creating composite rule."""
        rule = CompositeFieldRule(
            source_field="full_address",
            target_fields=["street_name", "city", "state", "zip_code"],
        )

        assert rule.source_field == "full_address"
        assert len(rule.target_fields) == 4


class TestMappingConfidence:
    """Test MappingConfidence enum."""

    def test_confidence_levels(self):
        """Test confidence level values."""
        assert MappingConfidence.HIGH.value == "high"
        assert MappingConfidence.MEDIUM.value == "medium"
        assert MappingConfidence.LOW.value == "low"

    def test_from_score_high(self):
        """Test high confidence from score."""
        level = MappingConfidence.from_score(0.95)
        assert level == MappingConfidence.HIGH

    def test_from_score_medium(self):
        """Test medium confidence from score."""
        level = MappingConfidence.from_score(0.75)
        assert level == MappingConfidence.MEDIUM

    def test_from_score_low(self):
        """Test low confidence from score."""
        level = MappingConfidence.from_score(0.5)
        assert level == MappingConfidence.LOW


class TestMappingCache:
    """Test MappingCache."""

    @pytest.fixture
    def temp_cache_path(self, tmp_path):
        """Create temp cache file path."""
        return str(tmp_path / "mapping_cache.json")

    @pytest.fixture
    def sample_mapping(self):
        """Create sample mapping."""
        return SchemaMapping(
            source_name="test_source",
            source_format="json",
            field_mappings=[
                FieldMapping("addr", "street_name", 0.9),
                FieldMapping("city", "city", 0.95),
            ],
            confidence=0.92,
        )

    @pytest.mark.asyncio
    async def test_initialize_empty_cache(self, temp_cache_path):
        """Test initializing empty cache."""
        cache = MappingCache(cache_path=temp_cache_path)
        await cache.initialize()

        assert cache.list_sources() == []
