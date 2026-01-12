"""ICDA utility modules."""

from .resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    retry_with_backoff,
    with_circuit_breaker,
    ValidationAuditEntry,
    ValidationAuditLog,
    validation_audit_log,
    sanitize_address_input,
)

from .zip_database import (
    ZipInfo,
    ZipDatabase,
    zip_database,
    build_zip_database_from_file,
)

__all__ = [
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "retry_with_backoff",
    "with_circuit_breaker",
    "ValidationAuditEntry",
    "ValidationAuditLog",
    "validation_audit_log",
    "sanitize_address_input",
    # ZIP Database
    "ZipInfo",
    "ZipDatabase",
    "zip_database",
    "build_zip_database_from_file",
]
