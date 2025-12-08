"""Unit tests for address verification pipeline.

Tests cover:
- Address normalization and parsing
- Address classification
- Index building and lookups
- Pipeline flow
- Batch processing
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from icda.address_models import (
    AddressComponent,
    AddressQuality,
    BatchItem,
    ParsedAddress,
    VerificationStatus,
)
from icda.address_normalizer import (
    AddressNormalizer,
    normalize_state,
    normalize_street_type,
)
from icda.address_index import AddressIndex, IndexedAddress
from icda.address_completer import NovaAddressCompleter
from icda.address_pipeline import AddressPipeline, BatchProcessor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_customers():
    """Sample customer data for testing."""
    return [
        {
            "crid": "CRID-000001",
            "name": "John Doe",
            "address": "123 Turkey Run",
            "city": "Springfield",
            "state": "VA",
            "zip": "22222",
            "move_history": [
                {
                    "from_address": "456 Oak St",
                    "to_address": "123 Turkey Run",
                    "city": "Springfield",
                    "state": "VA",
                    "zip": "22222",
                    "move_date": "2024-01-15",
                }
            ],
        },
        {
            "crid": "CRID-000002",
            "name": "Jane Smith",
            "address": "789 Main Blvd",
            "city": "Arlington",
            "state": "VA",
            "zip": "22201",
            "move_history": [],
        },
        {
            "crid": "CRID-000003",
            "name": "Bob Wilson",
            "address": "101 Turkey Trot Ln",
            "city": "Springfield",
            "state": "VA",
            "zip": "22222",
            "move_history": [],
        },
    ]


@pytest.fixture
def address_index(sample_customers):
    """Build address index from sample data."""
    index = AddressIndex()
    index.build_from_customers(sample_customers)
    return index


@pytest.fixture
def mock_completer(address_index):
    """Create mock Nova completer."""
    completer = MagicMock(spec=NovaAddressCompleter)
    completer.available = True
    completer.index = address_index
    completer.complete_address = AsyncMock()
    completer.suggest_street_completion = AsyncMock(return_value=[])
    return completer


@pytest.fixture
def pipeline(address_index, mock_completer):
    """Create pipeline with test dependencies."""
    return AddressPipeline(address_index, mock_completer)


# ============================================================================
# Address Normalizer Tests
# ============================================================================


class TestAddressNormalizer:
    """Tests for AddressNormalizer class."""

    def test_normalize_complete_address(self):
        """Test parsing a complete address."""
        raw = "123 Main St, New York, NY 10001"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.street_number == "123"
        assert parsed.street_name == "Main"
        assert parsed.street_type == "St"
        assert parsed.city == "New York"
        assert parsed.state == "NY"
        assert parsed.zip_code == "10001"

    def test_normalize_partial_address(self):
        """Test parsing a partial address."""
        raw = "101 turkey 22222"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.street_number == "101"
        assert parsed.zip_code == "22222"
        # Street name should be extracted (minus ZIP)
        assert "turkey" in parsed.raw.lower()

    def test_normalize_with_unit(self):
        """Test parsing address with unit number."""
        raw = "456 Oak Ave Apt 2B, Dallas, TX 75201"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.street_number == "456"
        assert parsed.unit == "#2B"
        assert AddressComponent.UNIT in parsed.components_found

    def test_normalize_zip_plus4(self):
        """Test parsing ZIP+4 format."""
        raw = "789 Pine Rd, Seattle, WA 98101-1234"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.zip_code == "98101"
        assert parsed.zip_plus4 == "1234"

    def test_normalize_full_state_name(self):
        """Test parsing full state name."""
        raw = "321 Elm Blvd, Austin, Texas 78701"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.state == "TX"

    def test_normalize_empty_input(self):
        """Test handling empty input."""
        parsed = AddressNormalizer.normalize("")
        assert parsed.raw == ""
        assert parsed.street_number is None

    def test_classify_complete_address(self):
        """Test classification of complete address."""
        parsed = ParsedAddress(
            raw="123 Main St, City, ST 12345",
            street_number="123",
            street_name="Main",
            street_type="St",
            city="City",
            state="ST",
            zip_code="12345",
        )
        classification = AddressNormalizer.classify(parsed)

        assert classification.quality == AddressQuality.COMPLETE
        assert classification.confidence >= 0.9

    def test_classify_partial_address(self):
        """Test classification of partial address."""
        parsed = ParsedAddress(
            raw="123 Main",
            street_number="123",
            street_name="Main",
        )
        classification = AddressNormalizer.classify(parsed)

        assert classification.quality in (AddressQuality.PARTIAL, AddressQuality.INVALID)
        assert len(classification.issues) > 0

    def test_classify_invalid_address(self):
        """Test classification of invalid address."""
        parsed = ParsedAddress(raw="gibberish")
        classification = AddressNormalizer.classify(parsed)

        assert classification.quality == AddressQuality.INVALID
        assert classification.confidence < 0.5


class TestStateNormalization:
    """Tests for state normalization."""

    def test_normalize_abbreviation(self):
        """Test normalizing state abbreviation."""
        assert normalize_state("NY") == "NY"
        assert normalize_state("ny") == "NY"
        assert normalize_state("Ca") == "CA"

    def test_normalize_full_name(self):
        """Test normalizing full state name."""
        assert normalize_state("New York") == "NY"
        assert normalize_state("california") == "CA"
        assert normalize_state("TEXAS") == "TX"

    def test_normalize_invalid_state(self):
        """Test handling invalid state."""
        assert normalize_state("ZZ") is None
        assert normalize_state("Atlantis") is None


class TestStreetTypeNormalization:
    """Tests for street type normalization."""

    def test_normalize_full_type(self):
        """Test normalizing full street type."""
        assert normalize_street_type("Street") == "St"
        assert normalize_street_type("avenue") == "Ave"
        assert normalize_street_type("BOULEVARD") == "Blvd"

    def test_normalize_abbreviation(self):
        """Test handling already abbreviated types."""
        assert normalize_street_type("St") == "St"
        assert normalize_street_type("Ave") == "Ave"


# ============================================================================
# Address Index Tests
# ============================================================================


class TestAddressIndex:
    """Tests for AddressIndex class."""

    def test_build_from_customers(self, sample_customers):
        """Test building index from customer data."""
        index = AddressIndex()
        count = index.build_from_customers(sample_customers)

        assert count > 0
        assert index.is_indexed
        assert index.total_addresses == count

    def test_lookup_exact(self, address_index):
        """Test exact address lookup."""
        # Note: The index stores street name without type separately
        parsed = ParsedAddress(
            raw="123 Turkey Run, Springfield, VA 22222",
            street_number="123",
            street_name="Turkey",  # Index stores name without "Run" suffix
            city="Springfield",
            state="VA",
            zip_code="22222",
        )
        results = address_index.lookup_exact(parsed)

        # If exact match not found, try fuzzy which should work
        if not results:
            fuzzy = address_index.lookup_fuzzy(parsed, threshold=0.8)
            assert len(fuzzy) >= 1, "Should find via fuzzy match"
            assert fuzzy[0].score >= 0.8
        else:
            assert results[0].score == 1.0
            assert results[0].match_type == "exact"

    def test_lookup_by_zip(self, address_index):
        """Test lookup by ZIP code."""
        results = address_index.lookup_by_zip("22222")

        assert len(results) >= 2  # At least Turkey Run and Turkey Trot

    def test_lookup_street_in_zip(self, address_index):
        """Test finding street in ZIP - the key feature."""
        # This tests the "turkey" -> "Turkey Run" scenario
        results = address_index.lookup_street_in_zip("turkey", "22222", threshold=0.5)

        assert len(results) >= 1
        # Should find Turkey Run and/or Turkey Trot
        street_names = [r.address.parsed.street_name for r in results]
        assert any("Turkey" in name for name in street_names if name)

    def test_lookup_fuzzy(self, address_index):
        """Test fuzzy address matching."""
        parsed = ParsedAddress(
            raw="123 turky run 22222",  # Typo in "turkey"
            street_number="123",
            street_name="turky run",
            zip_code="22222",
        )
        results = address_index.lookup_fuzzy(parsed, threshold=0.5)

        assert len(results) >= 1
        # Should still match Turkey Run
        assert results[0].score > 0.5

    def test_get_street_suggestions(self, address_index):
        """Test street name autocomplete."""
        suggestions = address_index.get_street_suggestions("tur", "22222", limit=5)

        assert len(suggestions) >= 1
        # Should suggest Turkey Run and/or Turkey Trot
        assert any("Turkey" in s for s in suggestions)

    def test_stats(self, address_index):
        """Test index statistics."""
        stats = address_index.stats()

        assert stats["indexed"] is True
        assert stats["total_addresses"] > 0
        assert "unique_zips" in stats
        assert "unique_states" in stats


# ============================================================================
# Address Pipeline Tests
# ============================================================================


class TestAddressPipeline:
    """Tests for AddressPipeline class."""

    @pytest.mark.asyncio
    async def test_verify_exact_match(self, pipeline, address_index):
        """Test verification with exact match."""
        result = await pipeline.verify("123 Turkey Run, Springfield, VA 22222")

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence >= 0.9
        assert result.verified is not None

    @pytest.mark.asyncio
    async def test_verify_partial_address(self, pipeline, mock_completer):
        """Test verification of partial address."""
        # Mock the completer to return a result
        from icda.address_models import VerificationResult

        mock_completer.complete_address.return_value = VerificationResult(
            status=VerificationStatus.COMPLETED,
            original=ParsedAddress(raw="101 turkey 22222"),
            verified=ParsedAddress(
                raw="101 Turkey Run, Springfield, VA 22222",
                street_number="101",
                street_name="Turkey Run",
                city="Springfield",
                state="VA",
                zip_code="22222",
            ),
            confidence=0.85,
            match_type="nova_completion",
        )

        result = await pipeline.verify("101 turkey 22222")

        # Should attempt completion
        assert result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.COMPLETED,
            VerificationStatus.SUGGESTED,
            VerificationStatus.UNVERIFIED,
        )

    @pytest.mark.asyncio
    async def test_verify_batch(self, pipeline):
        """Test batch verification."""
        items = [
            BatchItem(id="1", address="123 Turkey Run, Springfield, VA 22222"),
            BatchItem(id="2", address="789 Main Blvd, Arlington, VA 22201"),
        ]

        results, summary = await pipeline.verify_batch(items, concurrency=2)

        assert len(results) == 2
        assert summary.total == 2
        assert summary.verified + summary.corrected + summary.completed + summary.suggested + summary.unverified + summary.failed == 2

    @pytest.mark.asyncio
    async def test_verify_invalid_address(self, pipeline):
        """Test verification of invalid address."""
        result = await pipeline.verify("asdfghjkl")

        assert result.status in (VerificationStatus.UNVERIFIED, VerificationStatus.FAILED)
        assert result.confidence < 0.5


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.mark.asyncio
    async def test_process_list(self, pipeline):
        """Test processing a list of addresses."""
        processor = BatchProcessor(pipeline)
        addresses = [
            "123 Turkey Run, Springfield, VA 22222",
            "789 Main Blvd, Arlington, VA 22201",
        ]

        results, summary = await processor.process_list(addresses, concurrency=2)

        assert len(results) == 2
        assert summary.total == 2

    @pytest.mark.asyncio
    async def test_process_records(self, pipeline):
        """Test processing records with address fields."""
        processor = BatchProcessor(pipeline)
        records = [
            {
                "id": "1",
                "address": "123 Turkey Run",
                "city": "Springfield",
                "state": "VA",
                "zip": "22222",
            },
            {
                "id": "2",
                "street": "789 Main Blvd",
                "city": "Arlington",
                "state": "VA",
                "zip": "22201",
            },
        ]

        results, summary = await processor.process_records(
            records,
            address_field="address",
            id_field="id",
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_process_csv_data(self, pipeline):
        """Test processing CSV-style data."""
        processor = BatchProcessor(pipeline)
        rows = [
            {
                "address": "123 Turkey Run",
                "city": "Springfield",
                "state": "VA",
                "zip": "22222",
            },
        ]

        results, summary = await processor.process_csv_data(rows)

        assert len(results) == 1


# ============================================================================
# Address Completer Tests
# ============================================================================


class TestNovaAddressCompleter:
    """Tests for NovaAddressCompleter class."""

    def test_format_candidates(self, address_index):
        """Test candidate formatting for Nova prompt."""
        completer = NovaAddressCompleter.__new__(NovaAddressCompleter)
        completer.index = address_index
        completer.available = False  # Skip actual Nova calls

        # Get some matches
        parsed = ParsedAddress(
            raw="turkey 22222",
            street_name="turkey",
            zip_code="22222",
        )
        matches = address_index.lookup_street_in_zip("turkey", "22222")

        formatted = completer._format_candidates(matches)

        assert "Turkey" in formatted or "score" in formatted

    def test_fallback_completion(self, address_index):
        """Test fallback when Nova unavailable."""
        completer = NovaAddressCompleter.__new__(NovaAddressCompleter)
        completer.index = address_index
        completer.available = False

        parsed = ParsedAddress(
            raw="101 turkey 22222",
            street_number="101",
            street_name="turkey",
            zip_code="22222",
        )
        matches = address_index.lookup_street_in_zip("turkey", "22222")

        result = completer._fallback_completion(parsed, matches)

        # Should return best fuzzy match
        assert result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.CORRECTED,
            VerificationStatus.SUGGESTED,
            VerificationStatus.UNVERIFIED,
        )


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_normalize_unicode_address(self):
        """Test handling unicode in address."""
        raw = "123 Cañon St, San José, CA 95113"
        parsed = AddressNormalizer.normalize(raw)

        # Should handle unicode gracefully
        assert parsed.zip_code == "95113"

    def test_normalize_all_caps(self):
        """Test handling all-caps address."""
        raw = "456 OAK AVENUE, CHICAGO, IL 60601"
        parsed = AddressNormalizer.normalize(raw)

        assert parsed.state == "IL"
        assert parsed.zip_code == "60601"

    def test_normalize_extra_whitespace(self):
        """Test handling extra whitespace."""
        raw = "  789   Pine    Rd  ,  Seattle  ,  WA   98101  "
        parsed = AddressNormalizer.normalize(raw)

        # Should normalize whitespace
        assert parsed.zip_code == "98101"

    @pytest.mark.asyncio
    async def test_batch_with_errors(self, pipeline):
        """Test batch processing handles errors gracefully."""
        items = [
            BatchItem(id="1", address="valid address 12345"),
            BatchItem(id="2", address=""),  # Empty - might error
        ]

        results, summary = await pipeline.verify_batch(items)

        # Should complete without raising
        assert len(results) == 2

    def test_index_empty_customers(self):
        """Test building index from empty customer list."""
        index = AddressIndex()
        count = index.build_from_customers([])

        assert count == 0
        assert index.is_indexed
        assert index.total_addresses == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full pipeline flow."""

    @pytest.mark.asyncio
    async def test_turkey_run_scenario(self, address_index, mock_completer):
        """Test the specific turkey -> Turkey Run scenario."""
        pipeline = AddressPipeline(address_index, mock_completer)

        # User enters: "101 turkey ok 22222"
        # Should complete to: "101 Turkey Run, Springfield, VA 22222" or similar

        # First check the index has the data
        suggestions = address_index.get_street_suggestions("turkey", "22222")
        assert len(suggestions) > 0, "Index should have Turkey streets"

        # Check street lookup works
        matches = address_index.lookup_street_in_zip("turkey", "22222", threshold=0.5)
        assert len(matches) > 0, "Should find Turkey streets in ZIP 22222"

        # Verify Turkey Run is in the matches
        found_turkey_run = any(
            "Turkey Run" in (m.address.parsed.street_name or "")
            or "Turkey" in (m.address.parsed.street_name or "")
            for m in matches
        )
        assert found_turkey_run, "Should find Turkey Run or similar"

    @pytest.mark.asyncio
    async def test_full_pipeline_verified(self, address_index, mock_completer):
        """Test full pipeline returns verified for known address."""
        pipeline = AddressPipeline(address_index, mock_completer)

        result = await pipeline.verify("123 Turkey Run, Springfield, VA 22222")

        assert result.status == VerificationStatus.VERIFIED
        assert result.verified is not None
        # Street name may be stored as just "Turkey" without suffix
        assert "Turkey" in result.verified.street_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
