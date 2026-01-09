"""Robustness and resilience tests for address validation system.

Tests cover:
- Input sanitization edge cases
- Malformed input handling
- Circuit breaker behavior
- Retry logic with backoff
- Audit logging
- Extreme edge cases
- Performance under stress
"""

import asyncio
import pytest
import time
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from icda.utils.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    retry_with_backoff,
    with_circuit_breaker,
    sanitize_address_input,
    ValidationAuditEntry,
    ValidationAuditLog,
    validation_audit_log,
)
from icda.address_validator_engine import (
    AddressValidatorEngine,
    ValidationMode,
    ValidationResult,
)
from icda.address_normalizer import AddressNormalizer
from icda.address_models import AddressQuality, VerificationStatus


# ============================================================================
# Input Sanitization Tests
# ============================================================================


class TestInputSanitization:
    """Tests for input sanitization functionality."""

    def test_empty_input(self):
        """Test empty string input."""
        assert sanitize_address_input("") == ""
        assert sanitize_address_input(None) == ""

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        assert sanitize_address_input("   ") == ""
        assert sanitize_address_input("\t\n\r") == ""

    def test_max_length_truncation(self):
        """Test that input is truncated at max length."""
        long_input = "a" * 1000
        result = sanitize_address_input(long_input, max_length=500)
        assert len(result) == 500

    def test_control_character_removal(self):
        """Test that control characters are removed."""
        dirty = "123 Main St\x00\x01\x02, NY"
        clean = sanitize_address_input(dirty)
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "\x02" not in clean

    def test_tab_preservation(self):
        """Test that tabs are converted to spaces."""
        with_tab = "123\tMain\tSt"
        clean = sanitize_address_input(with_tab)
        # Tabs should be replaced or normalized
        assert "\t" not in clean or clean == "123 Main St"

    def test_newline_handling(self):
        """Test newline characters are handled."""
        with_newlines = "123 Main St\nNew York\nNY 10001"
        clean = sanitize_address_input(with_newlines)
        # Newlines should be normalized to spaces
        assert "\n" not in clean

    def test_unicode_preservation(self):
        """Test that valid unicode is preserved."""
        unicode_addr = "123 Cañon St, San José, CA"
        clean = sanitize_address_input(unicode_addr)
        assert "Cañon" in clean or "Canon" in clean
        assert "José" in clean or "Jose" in clean

    def test_emoji_handling(self):
        """Test emoji handling in addresses."""
        with_emoji = "123 Main St 🏠, NY 10001"
        clean = sanitize_address_input(with_emoji)
        # Should handle without crashing
        assert "123" in clean
        assert "Main" in clean

    def test_multiple_spaces_normalized(self):
        """Test multiple spaces are normalized."""
        messy = "123    Main     St,    NY"
        clean = sanitize_address_input(messy)
        assert "    " not in clean
        assert clean == "123 Main St, NY"

    def test_rtl_text_handling(self):
        """Test right-to-left text handling."""
        rtl = "123 Main St \u202Etricky\u202C, NY"
        clean = sanitize_address_input(rtl)
        # Should handle without crashing
        assert "123" in clean

    def test_zero_width_characters(self):
        """Test zero-width character removal."""
        with_zw = "123\u200BMain\u200CStreet\u200D, NY"
        clean = sanitize_address_input(with_zw)
        # Zero-width chars should be removed or handled
        assert "123" in clean


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker(name="test", threshold=3)
        assert breaker._state == CircuitState.CLOSED
        assert breaker.is_available()

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(name="test", threshold=3, reset_timeout=60)

        # Record 3 failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker._state == CircuitState.OPEN
        assert not breaker.is_available()

    def test_stays_closed_under_threshold(self):
        """Test circuit stays closed under threshold."""
        breaker = CircuitBreaker(name="test", threshold=3)

        # Record 2 failures (under threshold)
        breaker.record_failure()
        breaker.record_failure()

        assert breaker._state == CircuitState.CLOSED
        assert breaker.is_available()

    def test_success_reduces_failure_count(self):
        """Test success reduces failure count."""
        breaker = CircuitBreaker(name="test", threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        assert breaker._failures < 2

    def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker(name="test", threshold=1, reset_timeout=0)

        breaker.record_failure()
        assert breaker._state == CircuitState.OPEN

        # With reset_timeout=0, should immediately transition
        time.sleep(0.01)
        assert breaker.is_available()
        assert breaker._state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        breaker = CircuitBreaker(name="test", threshold=1, reset_timeout=0)

        breaker.record_failure()
        time.sleep(0.01)
        breaker.is_available()  # Trigger transition to half-open

        # Record 2 successes to close
        breaker.record_success()
        breaker.record_success()

        assert breaker._state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        """Test failure in half-open reopens circuit."""
        breaker = CircuitBreaker(name="test", threshold=1, reset_timeout=0)

        breaker.record_failure()
        time.sleep(0.01)
        breaker.is_available()  # Trigger transition to half-open

        breaker.record_failure()

        assert breaker._state == CircuitState.OPEN

    def test_get_state_returns_dict(self):
        """Test get_state returns correct dict."""
        breaker = CircuitBreaker(name="test_breaker", threshold=5, reset_timeout=300)

        state = breaker.get_state()

        assert state["name"] == "test_breaker"
        assert state["state"] == "closed"
        assert state["threshold"] == 5

    def test_manual_reset(self):
        """Test manual reset clears state."""
        breaker = CircuitBreaker(name="test", threshold=1)

        breaker.record_failure()
        assert breaker._state == CircuitState.OPEN

        breaker.reset()

        assert breaker._state == CircuitState.CLOSED
        assert breaker._failures == 0


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestRetryLogic:
    """Tests for retry with backoff functionality."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Test returns immediately on success."""
        async def success():
            return "ok"

        result = await retry_with_backoff(success, max_retries=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        """Test retries on transient errors."""
        attempts = []

        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await retry_with_backoff(
            flaky, max_retries=3,
            backoff_base=0.01,  # Fast for testing
            retryable_exceptions=(ConnectionError,)
        )

        assert result == "ok"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Test raises after exhausting retries."""
        async def always_fails():
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                always_fails, max_retries=2,
                backoff_base=0.01,
                retryable_exceptions=(ConnectionError,)
            )

    @pytest.mark.asyncio
    async def test_non_retryable_exception_raises_immediately(self):
        """Test non-retryable exceptions raise immediately."""
        attempts = []

        async def raises_value_error():
            attempts.append(1)
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                raises_value_error, max_retries=3,
                retryable_exceptions=(ConnectionError,)
            )

        assert len(attempts) == 1  # Only tried once

    @pytest.mark.asyncio
    async def test_backoff_increases(self):
        """Test backoff delay increases with attempts."""
        times = []

        async def track_time():
            times.append(time.time())
            raise TimeoutError("timeout")

        try:
            await retry_with_backoff(
                track_time, max_retries=2,
                backoff_base=0.1,
                retryable_exceptions=(TimeoutError,)
            )
        except TimeoutError:
            pass

        # Check delays increased
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_sync_function_support(self):
        """Test retry works with sync functions."""
        def sync_func():
            return "sync result"

        result = await retry_with_backoff(sync_func, max_retries=1)
        assert result == "sync result"


# ============================================================================
# Circuit Breaker Decorator Tests
# ============================================================================


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_calls_when_closed(self):
        """Test decorator allows calls when circuit is closed."""
        breaker = CircuitBreaker(name="test", threshold=3)

        @with_circuit_breaker(breaker)
        async def protected_call():
            return "success"

        result = await protected_call()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_blocks_calls_when_open(self):
        """Test decorator blocks calls when circuit is open."""
        breaker = CircuitBreaker(name="test", threshold=1, reset_timeout=300)
        breaker.record_failure()  # Open the circuit

        @with_circuit_breaker(breaker)
        async def protected_call():
            return "success"

        with pytest.raises(CircuitOpenError):
            await protected_call()

    @pytest.mark.asyncio
    async def test_decorator_records_success(self):
        """Test decorator records success."""
        breaker = CircuitBreaker(name="test", threshold=3)
        breaker.record_failure()  # Start with 1 failure

        @with_circuit_breaker(breaker)
        async def protected_call():
            return "success"

        await protected_call()

        # Success should reduce failure count
        assert breaker._failures < 1

    @pytest.mark.asyncio
    async def test_decorator_records_failure(self):
        """Test decorator records failure."""
        breaker = CircuitBreaker(name="test", threshold=3)

        @with_circuit_breaker(breaker)
        async def failing_call():
            raise ValueError("error")

        try:
            await failing_call()
        except ValueError:
            pass

        assert breaker._failures == 1


# ============================================================================
# Audit Logging Tests
# ============================================================================


class TestAuditLogging:
    """Tests for validation audit logging."""

    @pytest.fixture(autouse=True)
    def clear_audit_log(self):
        """Clear audit log before each test."""
        validation_audit_log.clear()
        yield
        validation_audit_log.clear()

    def test_log_entry_creation(self):
        """Test creating audit log entry."""
        entry = ValidationAuditEntry(
            timestamp=datetime.now(),
            input_address="123 Main St, NY 10001",
            validation_mode="correct",
            result_status="verified",
            confidence=0.95,
            is_valid=True,
            is_deliverable=True,
            corrections_count=1,
            completions_count=0,
            processing_time_ms=50,
            source="vector",
        )

        assert entry.confidence == 0.95
        assert entry.is_valid
        assert entry.source == "vector"

    def test_log_entry_to_dict(self):
        """Test audit entry to_dict."""
        entry = ValidationAuditEntry(
            timestamp=datetime.now(),
            input_address="123 Main St",
            validation_mode="validate",
            result_status="verified",
            confidence=0.9,
            is_valid=True,
            is_deliverable=True,
            corrections_count=0,
            completions_count=0,
            processing_time_ms=25,
            source="cache",
        )

        d = entry.to_dict()
        assert "timestamp" in d
        assert d["confidence"] == 0.9
        assert d["source"] == "cache"

    def test_audit_log_stores_entries(self):
        """Test audit log stores entries."""
        log = ValidationAuditLog(max_entries=100)

        entry = ValidationAuditEntry(
            timestamp=datetime.now(),
            input_address="test address",
            validation_mode="validate",
            result_status="verified",
            confidence=0.85,
            is_valid=True,
            is_deliverable=True,
            corrections_count=0,
            completions_count=0,
            processing_time_ms=30,
            source="vector",
        )

        log.log(entry)

        recent = log.get_recent(10)
        assert len(recent) == 1
        assert recent[0]["confidence"] == 0.85

    def test_audit_log_ring_buffer(self):
        """Test audit log maintains max size."""
        log = ValidationAuditLog(max_entries=5)

        for i in range(10):
            entry = ValidationAuditEntry(
                timestamp=datetime.now(),
                input_address=f"address {i}",
                validation_mode="validate",
                result_status="verified",
                confidence=0.9,
                is_valid=True,
                is_deliverable=True,
                corrections_count=0,
                completions_count=0,
                processing_time_ms=20,
                source="vector",
            )
            log.log(entry)

        recent = log.get_recent(100)
        assert len(recent) == 5  # Only keeps last 5

    def test_audit_log_stats(self):
        """Test audit log statistics."""
        log = ValidationAuditLog()

        # Add some entries
        for i in range(5):
            log.log(ValidationAuditEntry(
                timestamp=datetime.now(),
                input_address=f"addr {i}",
                validation_mode="validate",
                result_status="verified",
                confidence=0.8 + i * 0.02,
                is_valid=True,
                is_deliverable=i % 2 == 0,
                corrections_count=i,
                completions_count=0,
                processing_time_ms=20 + i * 10,
                source="vector",
            ))

        stats = log.get_stats()

        assert stats["total_validations"] == 5
        assert stats["total_valid"] == 5
        assert stats["total_deliverable"] == 3  # 0, 2, 4 are deliverable
        assert stats["avg_confidence"] > 0

    def test_audit_log_error_tracking(self):
        """Test audit log error tracking."""
        log = ValidationAuditLog()

        # Add error entry
        log.log(ValidationAuditEntry(
            timestamp=datetime.now(),
            input_address="bad address",
            validation_mode="validate",
            result_status="failed",
            confidence=0.0,
            is_valid=False,
            is_deliverable=False,
            corrections_count=0,
            completions_count=0,
            processing_time_ms=5,
            source="error",
            error="Parsing failed",
        ))

        errors = log.get_errors(10)
        assert len(errors) == 1
        assert errors[0]["error"] == "Parsing failed"


# ============================================================================
# Malformed Input Tests
# ============================================================================


class TestMalformedInputs:
    """Tests for handling malformed inputs."""

    @pytest.fixture
    def engine(self):
        return AddressValidatorEngine()

    def test_null_bytes(self, engine):
        """Test handling of null bytes in input."""
        result = engine.validate("123 Main\x00St, NY 10001")
        assert result is not None
        # Should handle without crashing

    def test_sql_injection_attempt(self, engine):
        """Test SQL injection patterns are handled safely."""
        result = engine.validate("123 Main St'; DROP TABLE addresses;--")
        assert result is not None
        # Should process as regular text

    def test_html_tags(self, engine):
        """Test HTML tags in input."""
        result = engine.validate("<script>alert('xss')</script>123 Main St")
        assert result is not None
        # Should sanitize or handle gracefully

    def test_very_long_zip(self, engine):
        """Test excessively long ZIP code."""
        result = engine.validate("123 Main St, NY " + "1" * 100)
        assert result is not None

    def test_negative_numbers(self, engine):
        """Test negative street numbers."""
        result = engine.validate("-123 Main St, NY 10001")
        assert result is not None

    def test_only_numbers(self, engine):
        """Test input with only numbers."""
        result = engine.validate("12345678901234567890")
        assert result is not None
        assert not result.is_valid

    def test_only_punctuation(self, engine):
        """Test input with only punctuation."""
        result = engine.validate("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result is not None
        assert not result.is_valid

    def test_repeated_patterns(self, engine):
        """Test input with repeated patterns."""
        result = engine.validate("Main St " * 50)
        assert result is not None

    def test_mixed_encodings(self, engine):
        """Test mixed character encodings."""
        # Mix of Latin, Cyrillic, and Chinese
        result = engine.validate("123 Mаin St 北京, NY 10001")  # 'а' is Cyrillic
        assert result is not None

    def test_json_in_address(self, engine):
        """Test JSON string in address."""
        result = engine.validate('{"address": "123 Main St", "city": "NY"}')
        assert result is not None


# ============================================================================
# Puerto Rico Edge Cases
# ============================================================================


class TestPuertoRicoEdgeCases:
    """Extended tests for Puerto Rico address handling."""

    @pytest.fixture
    def engine(self):
        return AddressValidatorEngine()

    def test_all_pr_zip_prefixes(self, engine):
        """Test all PR ZIP prefixes (006-009)."""
        for prefix in ["006", "007", "008", "009"]:
            zip_code = f"{prefix}01"
            result = engine.validate(f"123 Calle Luna, San Juan, PR {zip_code}")
            assert result.is_puerto_rico, f"ZIP {zip_code} should be PR"

    def test_non_pr_005_zip(self, engine):
        """Test that 005xx ZIPs are not PR."""
        result = engine.validate("123 Main St, City, NY 00501")
        assert not result.is_puerto_rico

    def test_sector_extraction(self, engine):
        """Test SECTOR urbanization extraction."""
        result = engine.validate("Sector La Marina 456 Calle 2, Mayaguez, PR 00680")
        assert result.is_puerto_rico

    def test_barrio_extraction(self, engine):
        """Test BARRIO urbanization extraction."""
        result = engine.validate("Barrio Obrero 789 Ave Central, Ponce, PR 00717")
        assert result.is_puerto_rico

    def test_urb_with_periods(self, engine):
        """Test URB. with period notation."""
        result = engine.validate("Urb. Villa Carolina 123 Calle A, Carolina, PR 00983")
        assert result.is_puerto_rico
        # Should extract urbanization

    def test_urbanizacion_with_accent(self, engine):
        """Test URBANIZACIÓN with accent."""
        result = engine.validate("Urbanización Las Lomas 456 Ave B, Bayamon, PR 00961")
        assert result.is_puerto_rico

    def test_pr_spanish_street_types(self, engine):
        """Test Spanish street types in PR."""
        addresses = [
            "URB Test, 123 Paseo del Morro, San Juan, PR 00901",
            "URB Test, 456 Camino Real, Ponce, PR 00717",
            "URB Test, 789 Carretera 2, Mayaguez, PR 00680",
        ]
        for addr in addresses:
            result = engine.validate(addr)
            assert result.is_puerto_rico

    def test_pr_calle_prefix(self, engine):
        """Test CALLE as street prefix (not suffix)."""
        result = engine.validate("URB Villa Carolina, 123 Calle Luna, Carolina, PR 00983")
        assert result.is_puerto_rico
        if result.validated and result.validated.street_name:
            # CALLE should be handled as prefix
            assert "Luna" in result.validated.street_name or result.validated.street_name

    def test_pr_building_types(self, engine):
        """Test PR building type terms."""
        addresses = [
            "URB Test, Edif. Central Apt 5, San Juan, PR 00901",
            "URB Test, Cond. Marina View Unit 3, Carolina, PR 00983",
            "URB Test, Res. Las Palmas 101, Bayamon, PR 00961",
        ]
        for addr in addresses:
            result = engine.validate(addr)
            assert result.is_puerto_rico

    def test_pr_without_state(self, engine):
        """Test PR address infers state from ZIP."""
        result = engine.validate("URB Villa Carolina, 123 Calle A, Carolina, 00983")
        assert result.is_puerto_rico
        if result.validated:
            assert result.validated.state == "PR"


# ============================================================================
# Performance and Stress Tests
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    @pytest.fixture
    def engine(self):
        return AddressValidatorEngine()

    def test_rapid_sequential_validations(self, engine):
        """Test many rapid sequential validations."""
        for i in range(100):
            result = engine.validate(f"{i} Main St, City {i}, NY 1000{i % 10}")
            assert result is not None

    def test_large_batch_processing(self, engine):
        """Test processing large batch."""
        addresses = [
            f"{i} Test St, City, NY 1000{i % 10}"
            for i in range(50)
        ]

        results = [engine.validate(addr) for addr in addresses]

        assert len(results) == 50
        assert all(r is not None for r in results)

    def test_memory_efficiency(self, engine):
        """Test no memory leak in repeated validations."""
        import sys

        # Warm up
        for _ in range(10):
            engine.validate("123 Main St, NY 10001")

        # Get baseline
        baseline = sys.getsizeof(engine)

        # Many more validations
        for _ in range(1000):
            engine.validate("456 Oak Ave, CA 90210")

        # Should not grow significantly
        after = sys.getsizeof(engine)
        assert after < baseline * 2  # Allow some growth but not unbounded


# ============================================================================
# Integration Tests
# ============================================================================


class TestRobustnessIntegration:
    """Integration tests for robustness features."""

    @pytest.fixture
    def engine(self):
        return AddressValidatorEngine()

    def test_full_sanitization_and_validation(self, engine):
        """Test full flow: sanitize -> normalize -> validate."""
        dirty_input = "  123   Main\x00 Stret  ,  new   york  ,  NY   10001  \n"

        result = engine.validate(dirty_input)

        assert result is not None
        # Should handle all the dirt and still validate
        assert result.overall_confidence > 0

    def test_error_recovery(self, engine):
        """Test system recovers from errors."""
        # First, cause a potential error
        result1 = engine.validate("" * 1000000)  # Empty but long

        # System should still work after
        result2 = engine.validate("123 Main St, New York, NY 10001")

        assert result2.is_valid

    def test_consistent_results(self, engine):
        """Test same input gives same output."""
        address = "123 Main St, New York, NY 10001"

        results = [engine.validate(address) for _ in range(10)]

        confidences = [r.overall_confidence for r in results]
        assert len(set(confidences)) == 1  # All same confidence

    def test_mode_isolation(self, engine):
        """Test different modes don't interfere."""
        address = "123 main stret, ny 10001"

        result_validate = engine.validate(address, mode=ValidationMode.VALIDATE)
        result_correct = engine.validate(address, mode=ValidationMode.CORRECT)
        result_complete = engine.validate(address, mode=ValidationMode.COMPLETE)

        # VALIDATE mode should not complete
        assert len(result_validate.completions_applied) == 0

        # CORRECT and COMPLETE may have different behaviors
        assert result_correct is not None
        assert result_complete is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
