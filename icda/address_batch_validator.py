"""Enhanced Batch Address Validator with Agent Orchestration.

This module provides high-quality batch address validation using the
5-agent orchestrator with quality enforcement gates and detailed
validation reporting.

Key Features:
- Uses agent orchestrator for intelligent address inference
- Quality gates enforcement per address
- Detailed validation results with issue categorization
- Concurrent processing with configurable parallelism
- Summary statistics with quality metrics
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from icda.address_models import ParsedAddress, VerificationStatus
from icda.agents.orchestrator import (
    AddressAgentOrchestrator,
    AgentResult,
    QualityGate,
    QualityGateResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION ISSUE CATEGORIES
# ============================================================================


class ValidationIssue(str, Enum):
    """Categories of address validation issues."""
    # Structural Issues
    MISSING_STREET = "missing_street"
    MISSING_CITY = "missing_city"
    MISSING_STATE = "missing_state"
    MISSING_ZIP = "missing_zip"
    INCOMPLETE_ZIP = "incomplete_zip"

    # Data Quality Issues
    INVALID_STATE = "invalid_state"
    INVALID_ZIP = "invalid_zip"
    STATE_CITY_MISMATCH = "state_city_mismatch"
    ZIP_STATE_MISMATCH = "zip_state_mismatch"

    # Format Issues
    TYPO_DETECTED = "typo_detected"
    CASE_INCONSISTENT = "case_inconsistent"
    EXTRA_WHITESPACE = "extra_whitespace"
    INVALID_UNIT_FORMAT = "invalid_unit_format"

    # Puerto Rico Specific
    MISSING_URBANIZATION = "missing_urbanization"
    PR_FORMAT_WARNING = "pr_format_warning"

    # Match Issues
    NO_MATCH_FOUND = "no_match_found"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_MATCHES = "multiple_matches"


@dataclass(slots=True)
class AddressValidationResult:
    """Result for a single address validation."""
    id: str
    original_address: str
    status: VerificationStatus
    confidence: float
    verified_address: ParsedAddress | None
    issues: list[ValidationIssue]
    quality_gates: list[QualityGateResult]
    corrections_made: list[str]
    alternatives_count: int
    processing_time_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_address": self.original_address,
            "status": self.status.value,
            "confidence": self.confidence,
            "verified_address": self.verified_address.to_dict() if self.verified_address else None,
            "issues": [i.value for i in self.issues],
            "quality_gates": [
                {"gate": g.gate.value, "passed": g.passed, "message": g.message}
                for g in self.quality_gates
            ],
            "corrections_made": self.corrections_made,
            "alternatives_count": self.alternatives_count,
            "processing_time_ms": self.processing_time_ms,
            "passed_all_gates": all(g.passed for g in self.quality_gates),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class BatchValidationSummary:
    """Summary statistics for batch validation."""
    total: int
    verified: int = 0
    corrected: int = 0
    suggested: int = 0
    unverified: int = 0
    failed: int = 0

    # Quality metrics
    avg_confidence: float = 0.0
    gates_pass_rate: float = 0.0

    # Issue breakdown
    issue_counts: dict[str, int] = field(default_factory=dict)

    # Timing
    total_time_ms: int = 0
    avg_time_ms: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (verified + corrected / total)."""
        if self.total == 0:
            return 0.0
        return (self.verified + self.corrected) / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "verified": self.verified,
            "corrected": self.corrected,
            "suggested": self.suggested,
            "unverified": self.unverified,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "gates_pass_rate": round(self.gates_pass_rate, 3),
            "issue_counts": self.issue_counts,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "top_issues": self._get_top_issues(5),
        }

    def _get_top_issues(self, n: int) -> list[dict[str, Any]]:
        """Get top N most common issues."""
        sorted_issues = sorted(
            self.issue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [{"issue": k, "count": v} for k, v in sorted_issues[:n]]


# ============================================================================
# AGENT BATCH VALIDATOR
# ============================================================================


class AgentBatchValidator:
    """High-quality batch address validator using agent orchestration.

    This validator:
    1. Uses the 5-agent orchestrator for intelligent verification
    2. Enforces quality gates per address
    3. Detects and categorizes validation issues
    4. Provides detailed corrections and alternatives
    5. Generates comprehensive quality metrics
    """

    # Valid US state codes (including territories)
    VALID_STATES = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC", "PR", "VI", "GU", "AS", "MP",
    }

    # State-city validation (partial - major cities)
    CITY_STATE_MAP = {
        "augusta": "GA", "baltimore": "MD", "oakland": "CA",
        "fresno": "CA", "mesa": "AZ", "laredo": "TX",
        "st. paul": "MN", "lincoln": "NE", "plano": "TX",
        "chandler": "AZ", "lexington": "KY", "riverside": "CA",
    }

    def __init__(self, orchestrator: AddressAgentOrchestrator):
        """Initialize with agent orchestrator.

        Args:
            orchestrator: Configured 5-agent orchestrator.
        """
        self._orchestrator = orchestrator

    async def validate_batch(
        self,
        addresses: list[str],
        concurrency: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[AddressValidationResult], BatchValidationSummary]:
        """Validate a batch of addresses using agent orchestration.

        Args:
            addresses: List of raw address strings.
            concurrency: Maximum concurrent validations.
            progress_callback: Optional callback(completed, total).

        Returns:
            Tuple of (results list, summary statistics).
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: list[AddressValidationResult] = []
        total = len(addresses)
        completed = 0
        start_time = time.time()

        async def process_address(idx: int, address: str) -> AddressValidationResult:
            nonlocal completed
            async with semaphore:
                item_start = time.time()
                try:
                    result = await self._validate_single(str(idx), address)
                    result.processing_time_ms = int((time.time() - item_start) * 1000)
                    return result
                except Exception as e:
                    logger.error(f"Validation error for '{address}': {e}")
                    return AddressValidationResult(
                        id=str(idx),
                        original_address=address,
                        status=VerificationStatus.FAILED,
                        confidence=0.0,
                        verified_address=None,
                        issues=[ValidationIssue.NO_MATCH_FOUND],
                        quality_gates=[],
                        corrections_made=[],
                        alternatives_count=0,
                        processing_time_ms=int((time.time() - item_start) * 1000),
                        metadata={"error": str(e)},
                    )
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        # Process all addresses concurrently
        tasks = [
            asyncio.create_task(process_address(i, addr))
            for i, addr in enumerate(addresses)
        ]
        results = await asyncio.gather(*tasks)

        # Calculate summary
        total_time = int((time.time() - start_time) * 1000)
        summary = self._calculate_summary(results, total_time)

        return results, summary

    async def _validate_single(
        self,
        item_id: str,
        address: str,
    ) -> AddressValidationResult:
        """Validate a single address with full issue detection.

        Args:
            item_id: Unique identifier for the address.
            address: Raw address string.

        Returns:
            Detailed validation result.
        """
        issues: list[ValidationIssue] = []
        corrections: list[str] = []

        # Pre-validation checks (before orchestrator)
        pre_issues = self._pre_validate(address)
        issues.extend(pre_issues)

        # Run through agent orchestrator
        agent_result, _ = await self._orchestrator.process(
            raw_address=address,
            session_history=None,
            hints=None,
            enable_trace=False,  # Skip trace for batch performance
        )

        # Analyze agent result for additional issues
        post_issues = self._analyze_result(address, agent_result)
        issues.extend(post_issues)

        # Detect corrections made
        if agent_result.verified_address:
            corrections = self._detect_corrections(
                address, agent_result.verified_address
            )

        # Remove duplicates while preserving order
        issues = list(dict.fromkeys(issues))

        return AddressValidationResult(
            id=item_id,
            original_address=address,
            status=agent_result.status,
            confidence=agent_result.confidence,
            verified_address=agent_result.verified_address,
            issues=issues,
            quality_gates=agent_result.quality_gates,
            corrections_made=corrections,
            alternatives_count=len(agent_result.alternatives),
            processing_time_ms=0,  # Set by caller
            metadata=agent_result.metadata,
        )

    def _pre_validate(self, address: str) -> list[ValidationIssue]:
        """Pre-validation checks on raw address."""
        issues = []

        # Check for extra whitespace
        if "  " in address or address != address.strip():
            issues.append(ValidationIssue.EXTRA_WHITESPACE)

        # Check case consistency
        words = address.split()
        if any(w.isupper() for w in words) and any(w.islower() for w in words):
            has_mixed = False
            for w in words:
                if len(w) > 2 and not w.isupper() and not w.islower() and not w.istitle():
                    has_mixed = True
                    break
            if has_mixed:
                issues.append(ValidationIssue.CASE_INCONSISTENT)

        # Check for incomplete ZIP (less than 5 digits at end)
        import re
        zip_match = re.search(r'\b(\d{1,4})$', address.strip())
        if zip_match and len(zip_match.group(1)) < 5:
            issues.append(ValidationIssue.INCOMPLETE_ZIP)

        # Check for common unit format issues
        if re.search(r'\bAnt\b', address, re.IGNORECASE):
            issues.append(ValidationIssue.INVALID_UNIT_FORMAT)

        return issues

    def _analyze_result(
        self,
        original: str,
        result: AgentResult,
    ) -> list[ValidationIssue]:
        """Analyze agent result for validation issues."""
        issues = []

        # Check quality gates for specific issues
        for gate in result.quality_gates:
            if not gate.passed:
                if gate.gate == QualityGate.PARSEABLE:
                    issues.append(ValidationIssue.MISSING_STREET)
                elif gate.gate == QualityGate.HAS_STREET:
                    issues.append(ValidationIssue.MISSING_STREET)
                elif gate.gate == QualityGate.HAS_LOCATION:
                    issues.append(ValidationIssue.MISSING_CITY)
                elif gate.gate == QualityGate.STATE_VALID:
                    issues.append(ValidationIssue.INVALID_STATE)
                elif gate.gate == QualityGate.ZIP_VALID:
                    issues.append(ValidationIssue.INVALID_ZIP)
                elif gate.gate == QualityGate.PR_URBANIZATION:
                    issues.append(ValidationIssue.MISSING_URBANIZATION)

        # Check confidence
        if result.confidence < 0.5:
            issues.append(ValidationIssue.LOW_CONFIDENCE)

        # Check for no match
        if result.status == VerificationStatus.UNVERIFIED:
            issues.append(ValidationIssue.NO_MATCH_FOUND)

        # Check for multiple alternatives (ambiguity)
        if len(result.alternatives) > 3:
            issues.append(ValidationIssue.MULTIPLE_MATCHES)

        # Check state-city mismatches
        if result.verified_address:
            addr = result.verified_address
            city_lower = (addr.city or "").lower()
            state = addr.state or ""

            expected_state = self.CITY_STATE_MAP.get(city_lower)
            if expected_state and expected_state != state:
                issues.append(ValidationIssue.STATE_CITY_MISMATCH)

        return issues

    def _detect_corrections(
        self,
        original: str,
        verified: ParsedAddress,
    ) -> list[str]:
        """Detect what corrections were made."""
        corrections = []
        original_lower = original.lower()

        # Check for typo corrections in street name
        if verified.street_name:
            street_lower = verified.street_name.lower()
            # Simple check: if street name not in original, likely corrected
            if street_lower not in original_lower:
                corrections.append(f"Street: {verified.street_name}")

        # Check for city corrections
        if verified.city:
            city_lower = verified.city.lower()
            if city_lower not in original_lower:
                corrections.append(f"City: {verified.city}")

        # Check for state standardization
        if verified.state:
            if verified.state not in original:
                corrections.append(f"State: {verified.state}")

        # Check for ZIP completion
        if verified.zip_code:
            if verified.zip_code not in original:
                corrections.append(f"ZIP: {verified.zip_code}")

        return corrections

    def _calculate_summary(
        self,
        results: list[AddressValidationResult],
        total_time_ms: int,
    ) -> BatchValidationSummary:
        """Calculate batch validation summary."""
        summary = BatchValidationSummary(
            total=len(results),
            total_time_ms=total_time_ms,
        )

        if not results:
            return summary

        # Count by status
        confidence_sum = 0.0
        gates_passed = 0
        gates_total = 0
        issue_counts: dict[str, int] = {}

        for r in results:
            # Status counts
            if r.status == VerificationStatus.VERIFIED:
                summary.verified += 1
            elif r.status == VerificationStatus.CORRECTED:
                summary.corrected += 1
            elif r.status == VerificationStatus.SUGGESTED:
                summary.suggested += 1
            elif r.status == VerificationStatus.UNVERIFIED:
                summary.unverified += 1
            else:
                summary.failed += 1

            # Confidence
            confidence_sum += r.confidence

            # Quality gates
            for gate in r.quality_gates:
                gates_total += 1
                if gate.passed:
                    gates_passed += 1

            # Issues
            for issue in r.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1

        # Calculate averages
        summary.avg_confidence = confidence_sum / len(results)
        summary.gates_pass_rate = gates_passed / gates_total if gates_total > 0 else 1.0
        summary.avg_time_ms = total_time_ms // len(results)
        summary.issue_counts = issue_counts

        return summary


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_batch_validator(
    orchestrator: AddressAgentOrchestrator,
) -> AgentBatchValidator:
    """Create a configured batch validator.

    Args:
        orchestrator: Configured 5-agent orchestrator.

    Returns:
        Ready-to-use batch validator.
    """
    return AgentBatchValidator(orchestrator)
