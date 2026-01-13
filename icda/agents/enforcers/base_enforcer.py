"""Base Enforcer - Abstract base class for memory enforcers.

All memory enforcers inherit from this class to provide
consistent interface and quality gate patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EnforcerGate(str, Enum):
    """Quality gates for memory enforcers."""
    # Memory Integrity Gates
    WRITE_CONSISTENCY = "write_consistency"
    EXTRACTION_COMPLETE = "extraction_complete"
    SESSION_ISOLATION = "session_isolation"
    ENTITY_COHERENCE = "entity_coherence"
    TTL_COMPLIANCE = "ttl_compliance"
    NAMESPACE_VALIDITY = "namespace_validity"

    # Search Context Gates
    CONTEXT_RELEVANCE = "context_relevance"
    FILTER_PRESERVATION = "filter_preservation"
    PRONOUN_RESOLUTION_ACCURACY = "pronoun_resolution_accuracy"
    LOCATION_CONTEXT_VALID = "location_context_valid"
    SEARCH_CONFIDENCE_STABLE = "search_confidence_stable"
    RESULT_CONSISTENCY = "result_consistency"

    # Nova Context Gates
    CONTEXT_TOKEN_BUDGET = "context_token_budget"
    RESPONSE_QUALITY_STABLE = "response_quality_stable"
    HALLUCINATION_PREVENTION = "hallucination_prevention"
    PERSONALITY_PRESERVATION = "personality_preservation"
    CONVERSATION_COHERENCE = "conversation_coherence"
    FACTUAL_GROUNDING = "factual_grounding"

    # Response Quality Gates
    OVERALL_QUALITY_MAINTAINED = "overall_quality_maintained"
    ALL_GATES_EVALUATED = "all_gates_evaluated"
    MEMORY_ATTRIBUTION = "memory_attribution"
    REGRESSION_DETECTION = "regression_detection"
    USER_SATISFACTION_PROXY = "user_satisfaction_proxy"
    COMPLETENESS_WITH_MEMORY = "completeness_with_memory"

    # Functionality Preservation Gates
    RESPONSE_PARITY = "response_parity"
    LATENCY_BUDGET = "latency_budget"
    FEATURE_COMPLETE = "feature_complete"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CONTEXT_PRESERVATION = "context_preservation"

    # RAG Context Gates
    RAG_CONTEXT_INCLUDED = "rag_context_included"
    RAG_CONFIDENCE_THRESHOLD = "rag_confidence_threshold"
    KNOWLEDGE_CHUNK_QUALITY = "knowledge_chunk_quality"
    CONTEXT_RELEVANCE_SCORE = "context_relevance_score"

    # Directory Coverage Gates
    DIRECTORY_COVERAGE_COMPLETE = "directory_coverage_complete"
    FILE_TYPE_SUPPORT = "file_type_support"
    INDEX_FRESHNESS = "index_freshness"
    ORPHAN_DETECTION = "orphan_detection"


@dataclass(slots=True)
class GateResult:
    """Result of a single gate check."""
    gate: EnforcerGate
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    threshold: float | None = None
    actual_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "gate": self.gate.value,
            "passed": self.passed,
            "message": self.message,
        }
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.actual_value is not None:
            result["actual_value"] = self.actual_value
        if self.details:
            result["details"] = self.details
        return result


@dataclass(slots=True)
class EnforcerResult:
    """Result from an enforcer validation.

    Attributes:
        enforcer_name: Name of the enforcer.
        passed: Whether all critical gates passed.
        gates_passed: List of gates that passed.
        gates_failed: List of gates that failed.
        quality_score: Overall quality score (0.0-1.0).
        recommendations: Suggested actions for failed gates.
        metrics: Performance and tracking metrics.
        modified_context: Modified context if enforcer made changes.
    """
    enforcer_name: str
    passed: bool = True
    gates_passed: list[GateResult] = field(default_factory=list)
    gates_failed: list[GateResult] = field(default_factory=list)
    quality_score: float = 1.0
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    modified_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enforcer_name": self.enforcer_name,
            "passed": self.passed,
            "gates_passed": [g.to_dict() for g in self.gates_passed],
            "gates_failed": [g.to_dict() for g in self.gates_failed],
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }

    @property
    def gates_count(self) -> int:
        """Total number of gates evaluated."""
        return len(self.gates_passed) + len(self.gates_failed)

    @property
    def pass_rate(self) -> float:
        """Percentage of gates that passed."""
        total = self.gates_count
        if total == 0:
            return 1.0
        return len(self.gates_passed) / total


class BaseEnforcer(ABC):
    """Abstract base class for memory enforcers.

    All memory enforcers must implement:
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
            name: Enforcer name for logging/tracking.
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the entire check.
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
    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run quality gates on the provided context.

        Args:
            context: Dictionary containing all data needed for validation.

        Returns:
            EnforcerResult with gate results and recommendations.
        """
        pass

    @abstractmethod
    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks.

        Returns:
            List of EnforcerGate values.
        """
        pass

    def _create_result(
        self,
        gates_passed: list[GateResult],
        gates_failed: list[GateResult],
    ) -> EnforcerResult:
        """Create an EnforcerResult from gate results.

        Args:
            gates_passed: List of passed gate results.
            gates_failed: List of failed gate results.

        Returns:
            EnforcerResult with calculated quality score.
        """
        total = len(gates_passed) + len(gates_failed)
        quality_score = len(gates_passed) / total if total > 0 else 1.0

        # In strict mode, any failure means overall failure
        passed = len(gates_failed) == 0 if self._strict_mode else quality_score >= 0.5

        recommendations = [
            f"Fix {g.gate.value}: {g.message}" for g in gates_failed
        ]

        return EnforcerResult(
            enforcer_name=self._name,
            passed=passed,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def _gate_pass(
        self,
        gate: EnforcerGate,
        message: str,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> GateResult:
        """Create a passing gate result."""
        return GateResult(
            gate=gate,
            passed=True,
            message=message,
            threshold=threshold,
            actual_value=actual_value,
            details=details or {},
        )

    def _gate_fail(
        self,
        gate: EnforcerGate,
        message: str,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> GateResult:
        """Create a failing gate result."""
        return GateResult(
            gate=gate,
            passed=False,
            message=message,
            threshold=threshold,
            actual_value=actual_value,
            details=details or {},
        )
