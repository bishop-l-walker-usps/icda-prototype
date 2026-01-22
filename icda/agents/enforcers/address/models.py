"""Shared models for Address Validation Enforcer Agents.

Dataclasses for metrics, reports, and configuration used by all
6 address validation enforcer agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AddressEnforcerGate(str, Enum):
    """Quality gates for address validation enforcers."""

    # Normalization Gates
    CASE_STANDARDIZED = "case_standardized"
    ABBREVIATIONS_EXPANDED = "abbreviations_expanded"
    WHITESPACE_NORMALIZED = "whitespace_normalized"
    PUNCTUATION_CLEANED = "punctuation_cleaned"
    DIRECTIONALS_STANDARDIZED = "directionals_standardized"
    UNIT_FORMAT_VALID = "unit_format_valid"

    # Completion Gates
    ZIP_INFERENCE_VALID = "zip_inference_valid"
    CITY_INFERENCE_VALID = "city_inference_valid"
    STATE_INFERENCE_VALID = "state_inference_valid"
    COMPLETION_CONFIDENCE = "completion_confidence"
    NO_CONFLICTING_INFERENCES = "no_conflicting_inferences"
    COMPONENT_CHAIN_VALID = "component_chain_valid"

    # Correction Gates
    TYPO_DETECTION_VALID = "typo_detection_valid"
    TRANSPOSITION_DETECTED = "transposition_detected"
    MISSPELLING_RECOVERED = "misspelling_recovered"
    CORRECTION_CONFIDENCE = "correction_confidence"
    NO_OVER_CORRECTION = "no_over_correction"
    ORIGINAL_INTENT_PRESERVED = "original_intent_preserved"

    # Match Confidence Gates
    CONFIDENCE_ABOVE_THRESHOLD = "confidence_above_threshold"
    ALTERNATIVES_SCORED = "alternatives_scored"
    NO_AMBIGUITY = "no_ambiguity"
    MATCH_TYPE_APPROPRIATE = "match_type_appropriate"
    MULTI_STATE_RESOLVED = "multi_state_resolved"
    COMPONENT_ALIGNMENT = "component_alignment"


class CorrectionType(str, Enum):
    """Types of corrections that can be applied to addresses."""

    TYPO = "typo"
    MISSPELLING = "misspelling"
    TRANSPOSITION = "transposition"
    ABBREVIATION = "abbreviation"
    CASE = "case"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    COMPLETION = "completion"
    FORMAT = "format"


class ComponentType(str, Enum):
    """Address component types."""

    STREET_NUMBER = "street_number"
    STREET_NAME = "street_name"
    STREET_TYPE = "street_type"
    UNIT_TYPE = "unit_type"
    UNIT_NUMBER = "unit_number"
    CITY = "city"
    STATE = "state"
    ZIP = "zip"
    ZIP_PLUS4 = "zip_plus4"
    DIRECTIONAL = "directional"
    URBANIZATION = "urbanization"


@dataclass(slots=True)
class NormalizationMetrics:
    """Metrics from normalization enforcer."""

    case_changes: int = 0
    abbreviations_expanded: int = 0
    whitespace_fixes: int = 0
    punctuation_fixes: int = 0
    directional_fixes: int = 0
    unit_format_fixes: int = 0

    original_text: str = ""
    normalized_text: str = ""
    changes_applied: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Total number of normalization changes."""
        return (
            self.case_changes
            + self.abbreviations_expanded
            + self.whitespace_fixes
            + self.punctuation_fixes
            + self.directional_fixes
            + self.unit_format_fixes
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_changes": self.case_changes,
            "abbreviations_expanded": self.abbreviations_expanded,
            "whitespace_fixes": self.whitespace_fixes,
            "punctuation_fixes": self.punctuation_fixes,
            "directional_fixes": self.directional_fixes,
            "unit_format_fixes": self.unit_format_fixes,
            "total_changes": self.total_changes,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "changes_applied": self.changes_applied,
        }


@dataclass(slots=True)
class CompletionMetrics:
    """Metrics from completion enforcer."""

    zip_inferred: bool = False
    city_inferred: bool = False
    state_inferred: bool = False
    inferred_zip: str = ""
    inferred_city: str = ""
    inferred_state: str = ""
    confidence: float = 0.0
    conflicts_found: int = 0
    conflict_details: list[str] = field(default_factory=list)
    inference_chain: list[str] = field(default_factory=list)

    @property
    def has_inferences(self) -> bool:
        """Check if any inferences were made."""
        return self.zip_inferred or self.city_inferred or self.state_inferred

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "zip_inferred": self.zip_inferred,
            "city_inferred": self.city_inferred,
            "state_inferred": self.state_inferred,
            "inferred_zip": self.inferred_zip,
            "inferred_city": self.inferred_city,
            "inferred_state": self.inferred_state,
            "confidence": self.confidence,
            "conflicts_found": self.conflicts_found,
            "conflict_details": self.conflict_details,
            "inference_chain": self.inference_chain,
        }


@dataclass(slots=True)
class CorrectionDetail:
    """Details of a single correction applied."""

    component: ComponentType
    original: str
    corrected: str
    correction_type: CorrectionType
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component.value,
            "original": self.original,
            "corrected": self.corrected,
            "correction_type": self.correction_type.value,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(slots=True)
class CorrectionMetrics:
    """Metrics from correction enforcer."""

    corrections: list[CorrectionDetail] = field(default_factory=list)
    typos_fixed: int = 0
    transpositions_fixed: int = 0
    misspellings_fixed: int = 0
    over_corrections: int = 0
    overall_confidence: float = 0.0
    intent_preserved: bool = True

    @property
    def total_corrections(self) -> int:
        """Total number of corrections."""
        return len(self.corrections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "corrections": [c.to_dict() for c in self.corrections],
            "typos_fixed": self.typos_fixed,
            "transpositions_fixed": self.transpositions_fixed,
            "misspellings_fixed": self.misspellings_fixed,
            "over_corrections": self.over_corrections,
            "overall_confidence": self.overall_confidence,
            "intent_preserved": self.intent_preserved,
            "total_corrections": self.total_corrections,
        }


@dataclass(slots=True)
class MatchConfidenceMetrics:
    """Metrics from match confidence enforcer."""

    primary_confidence: float = 0.0
    secondary_confidence: float = 0.0
    confidence_gap: float = 0.0
    ambiguity_score: float = 0.0
    match_type: str = ""
    alternatives_count: int = 0
    multi_state_candidates: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_ambiguous(self) -> bool:
        """Check if match is ambiguous (gap < 15%)."""
        return self.confidence_gap < 0.15 and self.secondary_confidence > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_confidence": self.primary_confidence,
            "secondary_confidence": self.secondary_confidence,
            "confidence_gap": self.confidence_gap,
            "ambiguity_score": self.ambiguity_score,
            "match_type": self.match_type,
            "alternatives_count": self.alternatives_count,
            "is_ambiguous": self.is_ambiguous,
            "multi_state_candidates": self.multi_state_candidates,
            "component_scores": self.component_scores,
        }


@dataclass(slots=True)
class AddressFixSummary:
    """Summary of all fixes applied to a single address."""

    address_id: int | str
    original_address: str
    final_address: str
    error_type: str = ""
    expected_correction: dict[str, Any] = field(default_factory=dict)

    # Enforcer results
    normalization: NormalizationMetrics | None = None
    completion: CompletionMetrics | None = None
    correction: CorrectionMetrics | None = None
    match_confidence: MatchConfidenceMetrics | None = None

    # Overall status
    is_valid: bool = False
    is_corrected: bool = False
    correction_successful: bool = False
    overall_confidence: float = 0.0
    quality_score: float = 0.0
    gates_passed: int = 0
    gates_failed: int = 0

    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address_id": self.address_id,
            "original_address": self.original_address,
            "final_address": self.final_address,
            "error_type": self.error_type,
            "expected_correction": self.expected_correction,
            "normalization": self.normalization.to_dict() if self.normalization else None,
            "completion": self.completion.to_dict() if self.completion else None,
            "correction": self.correction.to_dict() if self.correction else None,
            "match_confidence": self.match_confidence.to_dict() if self.match_confidence else None,
            "is_valid": self.is_valid,
            "is_corrected": self.is_corrected,
            "correction_successful": self.correction_successful,
            "overall_confidence": self.overall_confidence,
            "quality_score": self.quality_score,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ErrorTypeStats:
    """Statistics for a specific error type."""

    error_type: str
    total: int = 0
    validated: int = 0
    corrected: int = 0
    correction_successful: int = 0

    @property
    def validation_rate(self) -> float:
        """Validation success rate."""
        return self.validated / self.total if self.total > 0 else 0.0

    @property
    def correction_rate(self) -> float:
        """Correction application rate."""
        return self.corrected / self.total if self.total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Successful correction rate."""
        return self.correction_successful / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "total": self.total,
            "validated": self.validated,
            "corrected": self.corrected,
            "correction_successful": self.correction_successful,
            "validation_rate": self.validation_rate,
            "correction_rate": self.correction_rate,
            "success_rate": self.success_rate,
        }


@dataclass(slots=True)
class ComponentQualityStats:
    """Quality statistics for a specific address component."""

    component: str
    total_processed: int = 0
    successful: int = 0
    corrections_applied: int = 0
    errors_found: int = 0

    @property
    def success_rate(self) -> float:
        """Component success rate."""
        return self.successful / self.total_processed if self.total_processed > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "total_processed": self.total_processed,
            "successful": self.successful,
            "corrections_applied": self.corrections_applied,
            "errors_found": self.errors_found,
            "success_rate": self.success_rate,
        }


@dataclass(slots=True)
class BatchValidationReport:
    """Comprehensive report for batch address validation."""

    total_addresses: int = 0
    verified: int = 0
    corrected: int = 0
    failed: int = 0
    skipped: int = 0

    # Error type breakdown
    error_type_stats: dict[str, ErrorTypeStats] = field(default_factory=dict)

    # Component quality
    component_stats: dict[str, ComponentQualityStats] = field(default_factory=dict)

    # Enforcer summary
    normalization_changes: int = 0
    completions_applied: int = 0
    corrections_applied: int = 0
    avg_confidence: float = 0.0
    avg_quality_score: float = 0.0

    # Gate statistics
    total_gates_passed: int = 0
    total_gates_failed: int = 0

    # Timing
    total_time_ms: float = 0.0
    avg_time_per_address_ms: float = 0.0

    # Individual results
    address_summaries: list[AddressFixSummary] = field(default_factory=list)

    @property
    def verification_rate(self) -> float:
        """Overall verification rate."""
        return self.verified / self.total_addresses if self.total_addresses > 0 else 0.0

    @property
    def correction_rate(self) -> float:
        """Overall correction rate."""
        return self.corrected / self.total_addresses if self.total_addresses > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """Overall failure rate."""
        return self.failed / self.total_addresses if self.total_addresses > 0 else 0.0

    @property
    def gate_pass_rate(self) -> float:
        """Overall gate pass rate."""
        total = self.total_gates_passed + self.total_gates_failed
        return self.total_gates_passed / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_addresses": self.total_addresses,
            "verified": self.verified,
            "corrected": self.corrected,
            "failed": self.failed,
            "skipped": self.skipped,
            "verification_rate": self.verification_rate,
            "correction_rate": self.correction_rate,
            "failure_rate": self.failure_rate,
            "error_type_stats": {k: v.to_dict() for k, v in self.error_type_stats.items()},
            "component_stats": {k: v.to_dict() for k, v in self.component_stats.items()},
            "normalization_changes": self.normalization_changes,
            "completions_applied": self.completions_applied,
            "corrections_applied": self.corrections_applied,
            "avg_confidence": self.avg_confidence,
            "avg_quality_score": self.avg_quality_score,
            "total_gates_passed": self.total_gates_passed,
            "total_gates_failed": self.total_gates_failed,
            "gate_pass_rate": self.gate_pass_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_address_ms": self.avg_time_per_address_ms,
        }


@dataclass(slots=True)
class BatchProgress:
    """Real-time progress tracking for batch processing."""

    total: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    current_address: str = ""
    current_index: int = 0
    elapsed_ms: float = 0.0

    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @property
    def estimated_remaining_ms(self) -> float:
        """Estimated time remaining in ms."""
        if self.completed == 0:
            return 0.0
        avg_per_item = self.elapsed_ms / self.completed
        remaining = self.total - self.completed
        return avg_per_item * remaining


@dataclass(slots=True)
class BatchConfiguration:
    """Configuration for batch address validation."""

    concurrency: int = 10
    enable_normalization_enforcer: bool = True
    enable_completion_enforcer: bool = True
    enable_correction_enforcer: bool = True
    enable_match_confidence_enforcer: bool = True
    enable_report_generation: bool = True
    fail_fast: bool = False
    timeout_per_address_ms: int = 5000
    strict_mode: bool = False


@dataclass(slots=True)
class AddressInput:
    """Input address for validation."""

    id: int | str = 0
    address: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    error_type: str = ""

    # Additional expected fields for validation
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def full_address(self) -> str:
        """Build full address string."""
        parts = [self.address]
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.zip:
            parts.append(self.zip)
        return ", ".join(parts[:3]) + (f" {self.zip}" if self.zip else "")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "error_type": self.error_type,
            "full_address": self.full_address,
            "expected": self.expected,
        }
