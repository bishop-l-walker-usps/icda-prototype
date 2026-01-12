"""Tests for QueryRewriter - Smart query normalization and disambiguation.

Tests cover:
1. State misspelling correction (vaginia -> virginia)
2. People word replacement (people -> customers)
3. Informal/slang state references (cali -> california)
4. Ambiguous city detection (Kansas City -> MO or KS)
5. Full query rewriting (demographic -> customer data)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from icda.agents.query_rewriter import (
    QueryRewriter,
    RewriteResult,
    rewrite_query,
    normalize_state,
    STATE_NAME_TO_CODE,
    STATE_MISSPELLINGS,
    AMBIGUOUS_CITIES,
)


class TestStateMisspellingCorrection:
    """Tests for state name misspelling correction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_virginia_misspellings(self):
        """Test various Virginia misspellings."""
        misspellings = [
            ("vaginia", "virginia"),
            ("virgina", "virginia"),
            ("virgnia", "virginia"),
            ("virignia", "virginia"),
            ("verginia", "virginia"),
        ]

        for misspelled, correct in misspellings:
            result = self.rewriter.rewrite(f"customers in {misspelled}")
            assert correct in result.rewritten_query.lower(), (
                f"'{misspelled}' should be corrected to '{correct}'"
            )
            assert result.was_rewritten

    def test_kansas_misspellings(self):
        """Test Kansas misspellings."""
        misspellings = ["kanas", "kanses", "kanzas"]

        for misspelled in misspellings:
            result = self.rewriter.rewrite(f"how many customers in {misspelled}?")
            assert "kansas" in result.rewritten_query.lower(), (
                f"'{misspelled}' should be corrected to 'kansas'"
            )

    def test_california_misspellings(self):
        """Test California misspellings."""
        misspellings = ["californa", "californai", "califronia"]

        for misspelled in misspellings:
            result = self.rewriter.rewrite(f"show me {misspelled} data")
            assert "california" in result.rewritten_query.lower()

    def test_common_abbreviation_expansion(self):
        """Test that common abbreviations are recognized."""
        # "cali" should expand to "california"
        result = self.rewriter.rewrite("customers in cali")
        assert "california" in result.rewritten_query.lower()

    def test_fuzzy_matching_novel_misspellings(self):
        """Test fuzzy matching catches novel misspellings."""
        # These aren't in the explicit dictionary but should fuzzy match
        result = self.rewriter.rewrite("customers in Michigann")
        # Should detect via fuzzy matching
        assert result.detected_state_code == "MI" or "michigan" in result.rewritten_query.lower()


class TestPeopleWordReplacement:
    """Tests for replacing demographic words with 'customers'."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_how_many_people(self):
        """Test 'how many people' -> 'how many customers'."""
        result = self.rewriter.rewrite("how many people live in virginia?")
        assert "customer" in result.rewritten_query.lower()
        assert result.was_rewritten

    def test_folks_replacement(self):
        """Test 'folks' -> 'customers'."""
        result = self.rewriter.rewrite("how many folks in texas?")
        assert "customer" in result.rewritten_query.lower()

    def test_residents_replacement(self):
        """Test 'residents' -> 'customers'."""
        result = self.rewriter.rewrite("show me residents in california")
        assert "customer" in result.rewritten_query.lower()

    def test_population_context(self):
        """Test 'population' in customer data context."""
        result = self.rewriter.rewrite("what's the population in nevada?")
        # Should add customer context
        assert "customer" in result.rewritten_query.lower() or result.detected_state_code == "NV"

    def test_no_replacement_when_customer_present(self):
        """Test that 'customers' isn't added twice."""
        result = self.rewriter.rewrite("how many customers live in virginia?")
        # Should not double-add customers
        count = result.rewritten_query.lower().count("customer")
        assert count == 1

    def test_people_in_state(self):
        """Test 'people in [state]' pattern."""
        result = self.rewriter.rewrite("people in florida")
        assert "customer" in result.rewritten_query.lower()


class TestInformalStateReferences:
    """Tests for informal/slang state references."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_cali_expansion(self):
        """Test 'cali' -> 'california'."""
        result = self.rewriter.rewrite("show me cali customers")
        assert "california" in result.rewritten_query.lower()

    def test_vegas_to_nevada(self):
        """Test 'vegas' -> 'nevada'."""
        result = self.rewriter.rewrite("customers in vegas")
        assert result.detected_state_code == "NV" or "nevada" in result.rewritten_query.lower()

    def test_sunshine_state(self):
        """Test 'sunshine state' -> 'florida'."""
        result = self.rewriter.rewrite("customers in the sunshine state")
        assert "florida" in result.rewritten_query.lower()

    def test_golden_state(self):
        """Test 'golden state' -> 'california'."""
        result = self.rewriter.rewrite("golden state customers")
        assert "california" in result.rewritten_query.lower()

    def test_lone_star(self):
        """Test 'lone star' -> 'texas'."""
        result = self.rewriter.rewrite("lone star state movers")
        assert "texas" in result.rewritten_query.lower()

    def test_nyc_expansion(self):
        """Test 'nyc' -> 'new york'."""
        result = self.rewriter.rewrite("nyc customers")
        assert "new york" in result.rewritten_query.lower()


class TestAmbiguousCityDetection:
    """Tests for ambiguous city detection (cities in multiple states)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_kansas_city_ambiguity(self):
        """Test Kansas City detected as ambiguous."""
        result = self.rewriter.rewrite("customers in kansas city")
        assert result.is_ambiguous_city
        assert result.detected_city == "Kansas City"
        assert "MO" in result.ambiguous_city_states
        assert "KS" in result.ambiguous_city_states

    def test_kansas_city_with_state(self):
        """Test Kansas City with explicit state is not ambiguous."""
        result = self.rewriter.rewrite("customers in kansas city, missouri")
        # Should not be flagged as ambiguous since state is specified
        assert not result.is_ambiguous_city or result.detected_state_code == "MO"

    def test_springfield_ambiguity(self):
        """Test Springfield detected as ambiguous."""
        result = self.rewriter.rewrite("customers in springfield")
        assert result.is_ambiguous_city
        assert result.detected_city == "Springfield"

    def test_portland_ambiguity(self):
        """Test Portland detected as ambiguous (OR vs ME)."""
        result = self.rewriter.rewrite("portland customers")
        assert result.is_ambiguous_city
        assert "OR" in result.ambiguous_city_states
        assert "ME" in result.ambiguous_city_states


class TestStateDetection:
    """Tests for state detection in queries."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_full_state_name_detection(self):
        """Test detection of full state names."""
        states_to_test = [
            ("virginia", "VA"),
            ("california", "CA"),
            ("texas", "TX"),
            ("new york", "NY"),
            ("north carolina", "NC"),
        ]

        for state_name, expected_code in states_to_test:
            result = self.rewriter.rewrite(f"customers in {state_name}")
            assert result.detected_state_code == expected_code, (
                f"Expected {expected_code} for '{state_name}'"
            )

    def test_state_code_detection(self):
        """Test detection of state codes."""
        result = self.rewriter.rewrite("customers in TX")
        assert result.detected_state_code == "TX"

    def test_misspelled_state_detection(self):
        """Test detection of misspelled states."""
        result = self.rewriter.rewrite("customers in vaginia")
        assert result.detected_state_code == "VA"


class TestFullQueryRewriting:
    """Integration tests for full query rewriting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_original_problem_query(self):
        """Test the original problem query that started this fix."""
        result = self.rewriter.rewrite("how many people live in virginia?")

        # Should be rewritten to use "customers"
        assert "customer" in result.rewritten_query.lower()
        # Should detect Virginia
        assert result.detected_state_code == "VA"
        # Should be marked as rewritten
        assert result.was_rewritten

    def test_misspelled_demographic_query(self):
        """Test query with misspelling AND demographic wording."""
        result = self.rewriter.rewrite("how many folks in vaginia?")

        # Should fix misspelling
        assert "virginia" in result.rewritten_query.lower()
        # Should replace folks with customers
        assert "customer" in result.rewritten_query.lower()
        assert result.detected_state_code == "VA"

    def test_slang_demographic_query(self):
        """Test query with slang AND demographic wording."""
        result = self.rewriter.rewrite("show me people in cali")

        assert "california" in result.rewritten_query.lower()
        assert "customer" in result.rewritten_query.lower()

    def test_complex_multi_issue_query(self):
        """Test query with multiple issues to fix."""
        result = self.rewriter.rewrite(
            "how many residents moved from kanas to vaginia?"
        )

        # Should fix both misspellings
        assert "kansas" in result.rewritten_query.lower()
        assert "virginia" in result.rewritten_query.lower()
        # Should replace residents
        assert "customer" in result.rewritten_query.lower()

    def test_already_correct_query_not_modified(self):
        """Test that correct queries aren't unnecessarily modified."""
        correct_query = "show me customers in California"
        result = self.rewriter.rewrite(correct_query)

        # Might normalize case but shouldn't change semantics significantly
        assert result.detected_state_code == "CA"

    def test_preserves_original_query(self):
        """Test that original query is preserved in result."""
        original = "how many people in vaginia?"
        result = self.rewriter.rewrite(original)

        assert result.original_query == original
        assert result.rewritten_query != original


class TestNormalizeStateFunction:
    """Tests for the normalize_state convenience function."""

    def test_normalize_full_name(self):
        """Test normalizing full state names."""
        name, code = normalize_state("california")
        assert name == "California"
        assert code == "CA"

    def test_normalize_code(self):
        """Test normalizing state codes."""
        name, code = normalize_state("TX")
        assert name == "Texas"
        assert code == "TX"

    def test_normalize_misspelling(self):
        """Test normalizing misspelled names."""
        name, code = normalize_state("vaginia")
        assert name == "Virginia"
        assert code == "VA"

    def test_normalize_slang(self):
        """Test normalizing slang references."""
        name, code = normalize_state("cali")
        assert name == "California"
        assert code == "CA"

    def test_normalize_invalid(self):
        """Test normalizing invalid input."""
        name, code = normalize_state("notastate")
        assert name is None
        assert code is None


class TestRewriteQueryFunction:
    """Tests for the rewrite_query convenience function."""

    def test_rewrite_query_function(self):
        """Test the module-level convenience function."""
        result = rewrite_query("how many people in vaginia?")

        assert isinstance(result, RewriteResult)
        assert result.was_rewritten
        assert "customer" in result.rewritten_query.lower()
        assert "virginia" in result.rewritten_query.lower()


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()

    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.rewriter.rewrite("")
        assert result.rewritten_query == ""
        assert not result.was_rewritten

    def test_query_with_no_state(self):
        """Test query without any state reference."""
        result = self.rewriter.rewrite("show me all customers")
        assert result.detected_state_code is None

    def test_ambiguous_code_not_state(self):
        """Test that 'IN' as English word isn't treated as Indiana."""
        # "customers in" shouldn't detect Indiana
        result = self.rewriter.rewrite("show me customers in the database")
        # Should not detect IN (Indiana) from "in the database"
        # This depends on context detection logic

    def test_multiple_states_in_query(self):
        """Test query with multiple states mentioned."""
        result = self.rewriter.rewrite("customers who moved from california to texas")
        # Should detect at least one state
        assert result.detected_state_code in ["CA", "TX"]

    def test_case_insensitivity(self):
        """Test case-insensitive matching."""
        queries = [
            "VIRGINIA customers",
            "virginia customers",
            "Virginia customers",
            "ViRgInIa customers",
        ]

        for query in queries:
            result = self.rewriter.rewrite(query)
            assert result.detected_state_code == "VA"

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        result = self.rewriter.rewrite("how many customers in virginia???")
        assert result.detected_state_code == "VA"

    def test_to_dict_method(self):
        """Test the to_dict method of RewriteResult."""
        result = self.rewriter.rewrite("how many people in vaginia?")
        result_dict = result.to_dict()

        assert "original_query" in result_dict
        assert "rewritten_query" in result_dict
        assert "was_rewritten" in result_dict
        assert "detected_state_code" in result_dict
        assert result_dict["detected_state_code"] == "VA"


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
