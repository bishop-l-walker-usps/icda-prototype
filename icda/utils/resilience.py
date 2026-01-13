"""Resilience utilities for error-proof address validation.

Provides circuit breaker, retry logic, and audit logging for robust operation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar, Optional
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(name="nova", threshold=3, reset_timeout=300)

        if breaker.is_available():
            try:
                result = await call_external_service()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """
    name: str
    threshold: int = 3
    reset_timeout: int = 300  # seconds

    # Internal state
    _failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _success_count_half_open: int = field(default=0, init=False)

    def is_available(self) -> bool:
        """Check if circuit allows calls."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if enough time has passed to try again
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count_half_open = 0
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                return True
            return False

        # HALF_OPEN - allow limited calls
        return True

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count_half_open += 1
            # Need 2 successes to close circuit
            if self._success_count_half_open >= 2:
                self._state = CircuitState.CLOSED
                self._failures = 0
                logger.info(f"Circuit '{self.name}' CLOSED after recovery")
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            if self._failures > 0:
                self._failures = max(0, self._failures - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery test - back to OPEN
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' back to OPEN after failed recovery")
        elif self._failures >= self.threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit '{self.name}' OPENED after {self._failures} failures"
            )

    def get_state(self) -> dict[str, Any]:
        """Get circuit state for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._failures,
            "threshold": self.threshold,
            "seconds_until_retry": max(
                0,
                self.reset_timeout - (time.time() - self._last_failure_time)
            ) if self._state == CircuitState.OPEN else 0,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        logger.info(f"Circuit '{self.name}' manually reset")


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 2,
    backoff_base: float = 0.5,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, asyncio.TimeoutError),
    **kwargs,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        backoff_base: Base delay in seconds (multiplied by attempt number)
        retryable_exceptions: Tuple of exceptions that trigger retry

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = backoff_base * (attempt + 1)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                    f"after {type(e).__name__}: {e}. Waiting {delay}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                )

    raise last_exception


def with_circuit_breaker(breaker: CircuitBreaker):
    """Decorator to wrap a function with circuit breaker protection.

    Usage:
        nova_breaker = CircuitBreaker(name="nova")

        @with_circuit_breaker(nova_breaker)
        async def call_nova(prompt: str) -> str:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not breaker.is_available():
                raise CircuitOpenError(
                    f"Circuit '{breaker.name}' is open. "
                    f"Retry in {breaker.get_state()['seconds_until_retry']:.0f}s"
                )

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ============================================================================
# Audit Logging
# ============================================================================

@dataclass
class ValidationAuditEntry:
    """Single validation audit entry for tracking and debugging."""
    timestamp: datetime
    input_address: str
    validation_mode: str
    result_status: str
    confidence: float
    is_valid: bool
    is_deliverable: bool
    corrections_count: int
    completions_count: int
    processing_time_ms: int
    source: str  # "cache", "vector", "nova", "fallback"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "input_address": self.input_address[:100],  # Truncate for logs
            "validation_mode": self.validation_mode,
            "result_status": self.result_status,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "is_deliverable": self.is_deliverable,
            "corrections_count": self.corrections_count,
            "completions_count": self.completions_count,
            "processing_time_ms": self.processing_time_ms,
            "source": self.source,
            "error": self.error,
        }


class ValidationAuditLog:
    """Ring buffer audit log for validation operations.

    Keeps the last N validation entries for debugging and monitoring.
    """

    def __init__(self, max_entries: int = 1000):
        self._entries: deque[ValidationAuditEntry] = deque(maxlen=max_entries)
        self._stats = {
            "total_validations": 0,
            "total_valid": 0,
            "total_deliverable": 0,
            "total_errors": 0,
            "by_source": {"cache": 0, "vector": 0, "nova": 0, "fallback": 0, "error": 0},
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0,
        }

    def log(self, entry: ValidationAuditEntry) -> None:
        """Add an entry to the audit log."""
        self._entries.append(entry)
        self._update_stats(entry)

        # Also log to standard logger at DEBUG level
        logger.debug(
            f"AUDIT: {entry.input_address[:50]}... -> {entry.result_status} "
            f"({entry.confidence:.2f}) [{entry.source}] {entry.processing_time_ms}ms"
        )

    def _update_stats(self, entry: ValidationAuditEntry) -> None:
        """Update running statistics."""
        self._stats["total_validations"] += 1
        if entry.is_valid:
            self._stats["total_valid"] += 1
        if entry.is_deliverable:
            self._stats["total_deliverable"] += 1
        if entry.error:
            self._stats["total_errors"] += 1

        source = entry.source if entry.source in self._stats["by_source"] else "fallback"
        self._stats["by_source"][source] += 1

        # Running averages
        n = self._stats["total_validations"]
        self._stats["avg_confidence"] = (
            (self._stats["avg_confidence"] * (n - 1) + entry.confidence) / n
        )
        self._stats["avg_processing_time_ms"] = (
            (self._stats["avg_processing_time_ms"] * (n - 1) + entry.processing_time_ms) / n
        )

    def get_recent(self, count: int = 10) -> list[dict[str, Any]]:
        """Get the most recent entries."""
        entries = list(self._entries)[-count:]
        return [e.to_dict() for e in entries]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics."""
        total = self._stats["total_validations"]
        return {
            **self._stats,
            "valid_rate": self._stats["total_valid"] / total if total > 0 else 0,
            "deliverable_rate": self._stats["total_deliverable"] / total if total > 0 else 0,
            "error_rate": self._stats["total_errors"] / total if total > 0 else 0,
        }

    def get_errors(self, count: int = 10) -> list[dict[str, Any]]:
        """Get recent error entries."""
        errors = [e for e in self._entries if e.error]
        return [e.to_dict() for e in errors[-count:]]

    def clear(self) -> None:
        """Clear the audit log."""
        self._entries.clear()
        self._stats = {
            "total_validations": 0,
            "total_valid": 0,
            "total_deliverable": 0,
            "total_errors": 0,
            "by_source": {"cache": 0, "vector": 0, "nova": 0, "fallback": 0, "error": 0},
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0,
        }


# Global audit log instance
validation_audit_log = ValidationAuditLog()


# ============================================================================
# Input Sanitization
# ============================================================================

def sanitize_address_input(
    address: str,
    max_length: int = 500,
    strip_control_chars: bool = True,
) -> str:
    """Sanitize address input before processing.

    Args:
        address: Raw address input
        max_length: Maximum allowed length
        strip_control_chars: Remove control characters

    Returns:
        Sanitized address string
    """
    if not address:
        return ""

    # Truncate to max length
    result = address[:max_length]

    # Strip control characters (keep printable ASCII and common Unicode)
    if strip_control_chars:
        result = "".join(
            char for char in result
            if char.isprintable() or char in (" ", "\t")
        )

    # Normalize whitespace
    result = " ".join(result.split())

    return result.strip()
