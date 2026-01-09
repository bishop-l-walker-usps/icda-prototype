"""Base ingestion enforcer.

Abstract base class for ingestion pipeline enforcers,
mirroring the existing BaseEnforcer pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord


class IngestionGate(str, Enum):
    """Quality gates for address ingestion."""

    # Schema Enforcer Gates
    FIELDS_MAPPED = "fields_mapped"
    REQUIRED_PRESENT = "required_present"
    TYPES_VALID = "types_valid"
    SOURCE_ID_PRESENT = "source_id_present"

    # Normalization Enforcer Gates
    ADDRESS_PARSEABLE = "address_parseable"
    COMPONENTS_EXTRACTED = "components_extracted"
    STATE_NORMALIZED = "state_normalized"
    ZIP_FORMAT_VALID = "zip_format_valid"
    STREET_NAME_PRESENT = "street_name_present"

    # Duplicate Enforcer Gates
    NOT_IN_BATCH = "not_in_batch"
    NOT_IN_INDEX = "not_in_index"
    SIMILARITY_BELOW_THRESHOLD = "similarity_below_threshold"

    # Quality Enforcer Gates
    COMPLETENESS_SCORE = "completeness_score"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    PR_URBANIZATION = "pr_urbanization"
    NO_INVALID_CHARS = "no_invalid_chars"

    # Approval Enforcer Gates
    ALL_GATES_PASSED = "all_gates_passed"
    EMBEDDING_AVAILABLE = "embedding_available"
    INDEX_READY = "index_ready"
    QUALITY_THRESHOLD_MET = "quality_threshold_met"


@dataclass(slots=True)
class IngestionGateResult:
    """Result of a single gate check."""

    gate: IngestionGate
    passed: bool
    message: str
    score: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "gate": self.gate.value,
            "passed": self.passed,
            "message": self.message,
        }
        if self.score is not None:
            result["score"] = self.score
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.details:
            result["details"] = self.details
        return result


@dataclass(slots=True)
class IngestionEnforcerResult:
    """Result from an ingestion enforcer."""

    enforcer_name: str
    passed: bool = True
    gates_passed: list[IngestionGateResult] = field(default_factory=list)
    gates_failed: list[IngestionGateResult] = field(default_factory=list)
    quality_score: float = 1.0
    modified_record: IngestionRecord | None = None
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def gates_count(self) -> int:
        """Total gates evaluated."""
        return len(self.gates_passed) + len(self.gates_failed)

    @property
    def pass_rate(self) -> float:
        """Gate pass rate."""
        if self.gates_count == 0:
            return 1.0
        return len(self.gates_passed) / self.gates_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enforcer_name": self.enforcer_name,
            "passed": self.passed,
            "gates_passed": [g.to_dict() for g in self.gates_passed],
            "gates_failed": [g.to_dict() for g in self.gates_failed],
            "quality_score": self.quality_score,
            "pass_rate": self.pass_rate,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }


class BaseIngestionEnforcer(ABC):
    """Abstract base class for ingestion enforcers.

    Mirrors the existing BaseEnforcer pattern from
    icda/agents/enforcers/base_enforcer.py

    All ingestion enforcers must implement:
    - enforce(): Run quality gates and return result
    - get_gates(): Return list of gates this enforcer checks
    """

    __slots__ = ("_name", "_enabled", "_strict_mode")

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        strict_mode: bool = False,
    ):
        """Initialize enforcer.

        Args:
            name: Enforcer name for identification.
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the check.
        """
        self._name = name
        self._enabled = enabled
        self._strict_mode = strict_mode

    @property
    def name(self) -> str:
        """Get enforcer name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if enforcer is enabled."""
        return self._enabled

    @abstractmethod
    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Run quality gates on the record.

        Args:
            record: IngestionRecord to validate.
            context: Additional context (batch info, etc.).

        Returns:
            IngestionEnforcerResult with gate results.
        """
        pass

    @abstractmethod
    def get_gates(self) -> list[IngestionGate]:
        """Get list of gates this enforcer checks.

        Returns:
            List of IngestionGate values.
        """
        pass

    def _create_result(
        self,
        gates_passed: list[IngestionGateResult],
        gates_failed: list[IngestionGateResult],
        modified_record: IngestionRecord | None = None,
    ) -> IngestionEnforcerResult:
        """Create result from gate results.

        Args:
            gates_passed: Passed gate results.
            gates_failed: Failed gate results.
            modified_record: Modified record if any.

        Returns:
            IngestionEnforcerResult.
        """
        total = len(gates_passed) + len(gates_failed)
        quality_score = len(gates_passed) / total if total > 0 else 1.0

        # In strict mode, any failure = overall failure
        passed = (
            len(gates_failed) == 0
            if self._strict_mode
            else quality_score >= 0.5
        )

        recommendations = [
            f"Fix {g.gate.value}: {g.message}" for g in gates_failed
        ]

        return IngestionEnforcerResult(
            enforcer_name=self._name,
            passed=passed,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            quality_score=quality_score,
            modified_record=modified_record,
            recommendations=recommendations,
        )

    def _gate_pass(
        self,
        gate: IngestionGate,
        message: str,
        score: float | None = None,
        threshold: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> IngestionGateResult:
        """Create a passing gate result."""
        return IngestionGateResult(
            gate=gate,
            passed=True,
            message=message,
            score=score,
            threshold=threshold,
            details=details or {},
        )

    def _gate_fail(
        self,
        gate: IngestionGate,
        message: str,
        score: float | None = None,
        threshold: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> IngestionGateResult:
        """Create a failing gate result."""
        return IngestionGateResult(
            gate=gate,
            passed=False,
            message=message,
            score=score,
            threshold=threshold,
            details=details or {},
        )
