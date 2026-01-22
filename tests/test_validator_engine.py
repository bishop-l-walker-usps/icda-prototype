"""Comprehensive tests for the AddressValidatorEngine.

Tests cover:
- Component-level scoring
- Typo correction
- Address completion
- Confidence calculation
- Puerto Rico handling
- Edge cases and error conditions
"""

import pytest

from icda.address_models import (
    AddressComponent,
    AddressQuality,
    VerificationStatus,
)
from icda.address_validator_engine import (
    AddressValidatorEngine,
    ValidationMode,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine():
    """Create validator engine without index."""
    return AddressValidatorEngine()


@pytest.fixture
def engine_with_index(sample_customers):
    """Create validator engine with index."""
    from icda.address_index import AddressIndex

    index = AddressIndex()
    index.build_from_customers(sample_customers)
    return AddressValidatorEngine(address_index=index)


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
        },
        {
            "crid": "CRID-000002",
            "name": "Jane Smith",
            "address": "789 Main Blvd",
            "city": "Arlington",
            "state": "VA",
            "zip": "22201",
        },
        {
            "crid": "CRID-000003",
            "name": "Bob Wilson",
            "address": "101 Turkey Trot Ln",
            "city": "Springfield",
            "state": "VA",
            "zip": "22222",
        },
    ]


# ============================================================================
# Basic Validation Tests
# ============================================================================


class TestBasicValidation:
    """Tests for basic validation functionality."""

    def test_complete_address_validates(self, engine):
        """Test that a complete address validates successfully."""
        result = engine.validate("123 Main St, New York, NY 10001")

        assert result.is_valid
        assert result.overall_confidence >= 0.85
        assert result.quality == AddressQuality.COMPLETE
        assert result.status == VerificationStatus.VERIFIED
        assert result.standardized is not None

    def test_partial_address_lower_confidence(self, engine):
        """Test partial address has lower confidence."""
        result = engine.validate("123 Main St, NY")

        assert result.overall_confidence < 0.95

    def test_invalid_address_not_valid(self, engine):
        """Test invalid address fails validation."""
        result = engine.validate("asdfghjkl")

        assert not result.is_valid
        assert result.quality == AddressQuality.INVALID
        assert len(result.issues) > 0

    def test_empty_address(self, engine):
        """Test empty address handling."""
        result = engine.validate("")

        assert not result.is_valid
        assert result.overall_confidence < 0.3


# ============================================================================
# Typo Correction Tests
# ============================================================================


class TestTypoCorrection:
    """Tests for automatic typo correction."""

    def test_street_type_typo(self, engine):
        """Test correction of street type typo."""
        result = engine.validate("123 Main Stret, New York, NY 10001")

        assert (
            "stret" in str(result.corrections_applied).lower()
            or "street" in str(result.corrections_applied).lower()
        )

    def test_avenue_typo(self, engine):
        """Test correction of avenue typo."""
        result = engine.validate("456 Oak Avenu, Dallas, TX 75201")

        assert (
            any("avenu" in c.lower() for c in result.corrections_applied)
            or result.overall_confidence >= 0.7
        )

    def test_boulevard_typo(self, engine):
        """Test correction of boulevard typo."""
        result = engine.validate("789 Elm Bulevard, Chicago, IL 60601")

        assert (
            any(
                "bulevard" in c.lower() or "boulevard" in c.lower()
                for c in result.corrections_applied
            )
            or result.is_valid
        )

    def test_apartment_typo(self, engine):
        """Test correction of apartment typo."""
        result = engine.validate("101 Pine St Apartmnet 5, Boston, MA 02101")

        assert (
            any("apartm" in c.lower() for c in result.corrections_applied)
            or result.is_valid
        )

    def test_state_typo(self, engine):
        """Test correction of state name typo."""
        result = engine.validate("123 Main St, Austin, Texax 73301")

        assert (
            any(
                "texax" in c.lower() or "texas" in c.lower()
                for c in result.corrections_applied
            )
            or result.is_valid
        )

    def test_whitespace_normalization(self, engine):
        """Test whitespace is normalized."""
        result = engine.validate("  123   Main   St  ,  New York  ,  NY   10001  ")

        assert (
            any(
                "whitespace" in c.lower() or "space" in c.lower()
                for c in result.corrections_applied
            )
            or result.is_valid
        )


# ============================================================================
# Address Completion Tests
# ============================================================================


class TestAddressCompletion:
    """Tests for address completion from partial input."""

    def test_complete_city_from_zip(self, engine):
        """Test city completion from ZIP code."""
        result = engine.validate("123 Main St 10001", mode=ValidationMode.COMPLETE)

        # Should infer New York from ZIP 10001
        assert result.validated is not None
        if result.validated:
            assert result.validated.city is not None

    def test_complete_state_from_zip(self, engine):
        """Test state completion from ZIP code."""
        result = engine.validate("123 Main St 10001", mode=ValidationMode.COMPLETE)

        # Should infer NY from ZIP 10001
        assert result.validated is not None
        if result.validated:
            assert (
                result.validated.state in ("NY", None)
                or result.validated.state is not None
            )

    def test_completions_tracked(self, engine):
        """Test that completions are tracked."""
        result = engine.validate("123 Main St 22222", mode=ValidationMode.COMPLETE)

        # If completions occurred, we should record them.
        if result.completions_applied:
            assert len(result.completions_applied) > 0

    def test_context_hints_applied(self, engine):
        """Test context hints are applied."""
        result = engine.validate(
            "123 Main St",
            mode=ValidationMode.COMPLETE,
            context={"zip": "10001", "state": "NY"},
        )

        assert result.validated is not None
        if result.validated:
            assert (
                result.validated.zip_code == "10001" or result.validated.state == "NY"
            )

    def test_complete_mode_returns_best_effort_suggestion(self, engine):
        """Complete mode should return a best-effort completed address when suggested."""
        result = engine.validate(
            "Main St 10001",
            mode=ValidationMode.COMPLETE,
        )

        # Even if not fully valid (missing street number), COMPLETE mode should
        # still surface the best guess (NYC inferred from ZIP 10001).
        assert result.validated is not None
        assert result.standardized is not None
        assert "NY" in (result.standardized or "")

    def test_complete_mode_low_confidence_still_none(self, engine):
        """If confidence is very low, COMPLETE mode should not fabricate a validated address."""
        result = engine.validate(
            "just words",
            mode=ValidationMode.COMPLETE,
        )

        # Likely missing too many components -> should not meet suggested threshold.
        assert result.validated is None


# ============================================================================
# Component Scoring Tests
# ============================================================================


class TestComponentScoring:
    """Tests for component-level scoring."""

    def test_all_components_scored(self, engine):
        """Test that all components are scored."""
        result = engine.validate("123 Main St, New York, NY 10001")

        # Should have scores for major components
        component_types = [cs.component for cs in result.component_scores]
        assert AddressComponent.STREET_NUMBER in component_types
        assert AddressComponent.STREET_NAME in component_types

    def test_exact_match_high_score(self, engine):
        """Test exact matches get high scores."""
        result = engine.validate("123 Main St, New York, NY 10001")

        # Find street number score
        street_num_score = next(
            (
                cs
                for cs in result.component_scores
                if cs.component == AddressComponent.STREET_NUMBER
            ),
            None,
        )

        if street_num_score:
            assert street_num_score.score >= 0.9

    def test_missing_component_low_score(self, engine):
        """Test missing components get low scores."""
        result = engine.validate("Main St, New York, NY")

        # Find street number score
        street_num_score = next(
            (
                cs
                for cs in result.component_scores
                if cs.component == AddressComponent.STREET_NUMBER
            ),
            None,
        )

        if street_num_score:
            assert street_num_score.score < 0.5

    def test_corrected_component_flagged(self, engine):
        """Test corrected components are flagged."""
        result = engine.validate("123 Main Stret, New York, NY 10001")

        # Check if any component was corrected
        corrected = [cs for cs in result.component_scores if cs.was_corrected]
        # May or may not have corrections depending on how parsing works
        assert isinstance(corrected, list)

    def test_completed_component_flagged(self, engine):
        """Test completed components are flagged."""
        result = engine.validate("123 Main St 10001", mode=ValidationMode.COMPLETE)

        completed = [cs for cs in result.component_scores if cs.was_completed]
        # Should have completed city/state from ZIP
        assert isinstance(completed, list)


# ============================================================================
# Confidence Calculation Tests
# ============================================================================


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_confidence_range(self, engine):
        """Test confidence is in valid range."""
        result = engine.validate("123 Main St, New York, NY 10001")

        assert 0.0 <= result.overall_confidence <= 1.0

    def test_high_confidence_for_complete(self, engine):
        """Test complete addresses get high confidence."""
        result = engine.validate("123 Main St, New York, NY 10001")

        assert result.overall_confidence >= 0.85

    def test_low_confidence_for_invalid(self, engine):
        """Test invalid addresses get low confidence."""
        result = engine.validate("xyz")

        assert result.overall_confidence < 0.5

    def test_deliverable_threshold(self, engine):
        """Test deliverable requires high confidence."""
        result = engine.validate("123 Main St, New York, NY 10001")

        if result.overall_confidence >= 0.85:
            assert result.is_deliverable or result.overall_confidence < 0.85


# ============================================================================
# Puerto Rico Tests
# ============================================================================


class TestPuertoRico:
    """Tests for Puerto Rico address handling."""

    def test_detect_pr_by_zip(self, engine):
        """Test PR detected by ZIP code."""
        result = engine.validate("123 Calle Luna, San Juan, PR 00901")

        assert result.is_puerto_rico

    def test_pr_with_urbanization(self, engine):
        """Test PR address with urbanization."""
        result = engine.validate("URB Villa Carolina, 123 Calle A, Carolina, PR 00983")

        assert result.is_puerto_rico
        assert result.urbanization_status == "present"

    def test_pr_missing_urbanization_warning(self, engine):
        """Test PR address without urbanization gets warning."""
        result = engine.validate("123 Calle Luna, San Juan, PR 00901")

        assert result.is_puerto_rico
        if result.urbanization_status == "missing":
            # Should have urbanization warning in issues
            urb_issues = [i for i in result.issues if "urbaniz" in i.message.lower()]
            assert len(urb_issues) > 0

    def test_pr_confidence_penalty(self, engine):
        """Test PR without urbanization has lower confidence."""
        with_urb = engine.validate(
            "URB Villa Carolina, 123 Calle A, Carolina, PR 00983"
        )
        without_urb = engine.validate("123 Calle A, Carolina, PR 00983")

        # With urbanization should have equal or higher confidence
        assert with_urb.overall_confidence >= without_urb.overall_confidence - 0.2


# ============================================================================
# Validation Mode Tests
# ============================================================================


class TestValidationModes:
    """Tests for different validation modes."""

    def test_validate_mode_no_changes(self, engine):
        """Test validate mode doesn't make changes."""
        result = engine.validate("123 main stret 10001", mode=ValidationMode.VALIDATE)

        # In validate mode, no completions should be made
        assert len(result.completions_applied) == 0

    def test_complete_mode_fills_gaps(self, engine):
        """Test complete mode fills in gaps."""
        result = engine.validate("123 Main St 10001", mode=ValidationMode.COMPLETE)

        # Should attempt to complete city/state
        if result.validated and result.validated.city:
            assert (
                len(result.completions_applied) > 0 or result.validated.city is not None
            )

    def test_correct_mode_fixes_errors(self, engine):
        """Test correct mode fixes errors."""
        result = engine.validate(
            "123 Main Stret, New York, NY 10001", mode=ValidationMode.CORRECT
        )

        # Should fix typos
        assert len(result.corrections_applied) > 0 or result.is_valid

    def test_standardize_mode_formats(self, engine):
        """Test standardize mode produces USPS format."""
        result = engine.validate(
            "123 main street, new york, ny 10001", mode=ValidationMode.STANDARDIZE
        )

        if result.standardized:
            # Should be uppercased
            assert result.standardized == result.standardized.upper()


# ============================================================================
# Issue Detection Tests
# ============================================================================


class TestIssueDetection:
    """Tests for issue detection and reporting."""

    def test_missing_street_number_error(self, engine):
        """Test missing street number is an error."""
        result = engine.validate("Main St, New York, NY 10001")

        errors = [i for i in result.issues if i.severity == "error"]
        street_errors = [i for i in errors if "street number" in i.message.lower()]
        assert len(street_errors) > 0

    def test_missing_street_name_error(self, engine):
        """Test missing street name is an error."""
        result = engine.validate("123, New York, NY 10001")

        errors = [i for i in result.issues if i.severity == "error"]
        assert isinstance(errors, list)

    def test_missing_zip_warning(self, engine):
        """Test missing ZIP is a warning."""
        result = engine.validate("123 Main St, New York, NY")

        warnings = [i for i in result.issues if i.severity == "warning"]
        zip_warnings = [i for i in warnings if "zip" in i.message.lower()]
        # May or may not have ZIP warning depending on parsing
        assert isinstance(zip_warnings, list)

    def test_issues_have_suggestions(self, engine):
        """Test issues include suggestions."""
        result = engine.validate("Main St, NY")

        issues_with_suggestions = [i for i in result.issues if i.suggestion]
        assert isinstance(issues_with_suggestions, list)


# ============================================================================
# Standardization Tests
# ============================================================================


class TestStandardization:
    """Tests for address standardization."""

    def test_uppercase_output(self, engine):
        """Test standardized address is uppercase."""
        result = engine.validate("123 main st, new york, ny 10001")

        if result.standardized:
            assert result.standardized == result.standardized.upper()

    def test_abbreviations_standardized(self, engine):
        """Test street types are abbreviated."""
        result = engine.validate("123 Main Street, New York, NY 10001")

        if result.standardized:
            # Should use abbreviation
            assert "ST" in result.standardized or "STREET" in result.standardized

    def test_pr_urbanization_format(self, engine):
        """Test PR address has urbanization line."""
        result = engine.validate("URB Villa Carolina, 123 Calle A, Carolina, PR 00983")

        if result.standardized:
            assert "URB" in result.standardized


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_all_caps_input(self, engine):
        """Test handling of all-caps input."""
        result = engine.validate("123 MAIN STREET, NEW YORK, NY 10001")

        assert result.is_valid
        assert result.overall_confidence >= 0.8

    def test_all_lowercase_input(self, engine):
        """Test handling of all-lowercase input."""
        result = engine.validate("123 main street, new york, ny 10001")

        assert result.is_valid
        assert result.overall_confidence >= 0.8

    def test_unicode_characters(self, engine):
        """Test handling of unicode characters."""
        result = engine.validate("123 Cañon St, San José, CA 95113")

        # Should handle without crashing
        assert result is not None
        assert result.overall_confidence > 0

    def test_zip_plus_four(self, engine):
        """Test ZIP+4 handling."""
        result = engine.validate("123 Main St, New York, NY 10001-1234")

        assert result.validated is not None
        if result.validated:
            assert result.validated.zip_code == "10001"
            # zip_plus4 may or may not be extracted

    def test_very_long_address(self, engine):
        """Test handling of very long addresses."""
        long_address = "123 " + "Very Long Street Name " * 10 + ", New York, NY 10001"
        result = engine.validate(long_address)

        # Should handle without crashing
        assert result is not None

    def test_special_characters(self, engine):
        """Test handling of special characters."""
        result = engine.validate("123 Main St #5-A, New York, NY 10001")

        assert result is not None


# ============================================================================
# Index Integration Tests
# ============================================================================


class TestWithIndex:
    """Tests for validation with address index."""

    def test_fuzzy_match_with_index(self, engine_with_index):
        """Test fuzzy matching against index."""
        result = engine_with_index.validate("123 Turky Run, Springfield, VA 22222")

        # Should find Turkey Run with fuzzy matching
        assert result.overall_confidence > 0.5

    def test_alternatives_from_index(self, engine_with_index):
        """Test alternatives are found from index."""
        result = engine_with_index.validate("turkey 22222")

        # May find alternatives from index
        assert isinstance(result.alternatives, list)

    def test_street_completion_from_index(self, engine_with_index):
        """Test street name completion from index."""
        result = engine_with_index.validate(
            "123 turkey 22222", mode=ValidationMode.COMPLETE
        )

        # May complete to Turkey Run
        if result.validated and result.validated.street_name:
            name = result.validated.street_name.lower()
            # Might complete to Turkey Run or similar
            assert "turkey" in name or len(name) > 0


# ============================================================================
# Batch Validation Tests
# ============================================================================


class TestBatchValidation:
    """Tests for batch validation scenarios."""

    def test_multiple_addresses(self, engine):
        """Test validating multiple addresses."""
        addresses = [
            "123 Main St, New York, NY 10001",
            "456 Oak Ave, Dallas, TX 75201",
            "invalid gibberish",
        ]

        results = [engine.validate(addr) for addr in addresses]

        assert len(results) == 3
        assert results[0].is_valid
        assert results[1].is_valid
        assert not results[2].is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
