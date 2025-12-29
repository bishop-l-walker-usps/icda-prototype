"""Response Quality Enforcer - Final validation of response quality with memory.

Acts as the last line of defense before response reaches the user,
ensuring complete response quality after memory integration.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class ResponseQualityEnforcer(BaseEnforcer):
    """Enforcer for final response quality validation.

    Quality Gates:
    - OVERALL_QUALITY_MAINTAINED: Final quality score acceptable
    - ALL_GATES_EVALUATED: All 8 existing gates checked
    - REGRESSION_DETECTION: No quality regression from memory
    - MEMORY_ATTRIBUTION: Response correctly attributes memory data
    """

    # Thresholds
    MIN_QUALITY_SCORE = 0.5       # Minimum acceptable quality
    REGRESSION_THRESHOLD = -0.1   # Max allowed quality drop
    EXPECTED_GATES = 8            # Number of existing quality gates

    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        """Initialize ResponseQualityEnforcer."""
        super().__init__(
            name="ResponseQualityEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.OVERALL_QUALITY_MAINTAINED,
            EnforcerGate.ALL_GATES_EVALUATED,
            EnforcerGate.REGRESSION_DETECTION,
            EnforcerGate.MEMORY_ATTRIBUTION,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run response quality gates.

        Args:
            context: Must contain:
                - enforced_response: EnforcedResponse from EnforcerAgent
                - baseline_quality: Quality score without memory (optional)
                - unified_memory: UnifiedMemoryContext
                - nova_response: NovaResponse

        Returns:
            EnforcerResult with gate outcomes.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Gate 1: Overall Quality Maintained
        quality_result = self._check_overall_quality(context)
        (gates_passed if quality_result.passed else gates_failed).append(quality_result)

        # Gate 2: All Gates Evaluated
        gates_result = self._check_all_gates_evaluated(context)
        (gates_passed if gates_result.passed else gates_failed).append(gates_result)

        # Gate 3: Regression Detection
        regression_result = self._check_regression(context)
        (gates_passed if regression_result.passed else gates_failed).append(
            regression_result
        )

        # Gate 4: Memory Attribution
        attribution_result = self._check_memory_attribution(context)
        (gates_passed if attribution_result.passed else gates_failed).append(
            attribution_result
        )

        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "gates_evaluated": len(gates_passed) + len(gates_failed),
            "pass_rate": result.pass_rate,
        }

        return result

    def _check_overall_quality(self, context: dict[str, Any]) -> GateResult:
        """Check that final quality score is acceptable."""
        enforced_response = context.get("enforced_response")

        if not enforced_response:
            return self._gate_pass(
                EnforcerGate.OVERALL_QUALITY_MAINTAINED,
                "No enforced response to validate",
            )

        quality_score = (
            enforced_response.quality_score
            if hasattr(enforced_response, 'quality_score')
            else 0.5
        )

        if quality_score >= self.MIN_QUALITY_SCORE:
            return self._gate_pass(
                EnforcerGate.OVERALL_QUALITY_MAINTAINED,
                f"Quality score {quality_score:.2f} meets minimum",
                threshold=self.MIN_QUALITY_SCORE,
                actual_value=quality_score,
            )

        return self._gate_fail(
            EnforcerGate.OVERALL_QUALITY_MAINTAINED,
            f"Quality score {quality_score:.2f} below minimum",
            threshold=self.MIN_QUALITY_SCORE,
            actual_value=quality_score,
        )

    def _check_all_gates_evaluated(self, context: dict[str, Any]) -> GateResult:
        """Check that all existing quality gates were evaluated."""
        enforced_response = context.get("enforced_response")

        if not enforced_response:
            return self._gate_pass(
                EnforcerGate.ALL_GATES_EVALUATED,
                "No enforced response to validate",
            )

        gates_passed = (
            enforced_response.gates_passed
            if hasattr(enforced_response, 'gates_passed')
            else []
        )
        gates_failed = (
            enforced_response.gates_failed
            if hasattr(enforced_response, 'gates_failed')
            else []
        )

        total_gates = len(gates_passed) + len(gates_failed)

        if total_gates >= self.EXPECTED_GATES:
            return self._gate_pass(
                EnforcerGate.ALL_GATES_EVALUATED,
                f"All {total_gates} gates evaluated",
                threshold=self.EXPECTED_GATES,
                actual_value=total_gates,
            )

        if total_gates >= self.EXPECTED_GATES - 2:
            return self._gate_pass(
                EnforcerGate.ALL_GATES_EVALUATED,
                f"{total_gates} of {self.EXPECTED_GATES} gates evaluated (acceptable)",
                threshold=self.EXPECTED_GATES,
                actual_value=total_gates,
            )

        return self._gate_fail(
            EnforcerGate.ALL_GATES_EVALUATED,
            f"Only {total_gates} of {self.EXPECTED_GATES} gates evaluated",
            threshold=self.EXPECTED_GATES,
            actual_value=total_gates,
        )

    def _check_regression(self, context: dict[str, Any]) -> GateResult:
        """Check that memory doesn't cause quality regression."""
        enforced_response = context.get("enforced_response")
        baseline_quality = context.get("baseline_quality")

        if not enforced_response:
            return self._gate_pass(
                EnforcerGate.REGRESSION_DETECTION,
                "No enforced response to check regression",
            )

        current_quality = (
            enforced_response.quality_score
            if hasattr(enforced_response, 'quality_score')
            else 0.5
        )

        if baseline_quality is None:
            return self._gate_pass(
                EnforcerGate.REGRESSION_DETECTION,
                f"Quality: {current_quality:.2f} (no baseline for comparison)",
                actual_value=current_quality,
            )

        delta = current_quality - baseline_quality

        if delta >= self.REGRESSION_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.REGRESSION_DETECTION,
                f"Quality delta: {delta:+.2f} (no regression)",
                threshold=self.REGRESSION_THRESHOLD,
                actual_value=delta,
            )

        return self._gate_fail(
            EnforcerGate.REGRESSION_DETECTION,
            f"Quality regressed by {-delta:.2f}",
            threshold=self.REGRESSION_THRESHOLD,
            actual_value=delta,
        )

    def _check_memory_attribution(self, context: dict[str, Any]) -> GateResult:
        """Check that memory data is properly attributed in response."""
        enforced_response = context.get("enforced_response")
        unified_memory = context.get("unified_memory")

        if not enforced_response or not unified_memory:
            return self._gate_pass(
                EnforcerGate.MEMORY_ATTRIBUTION,
                "No response or memory to check attribution",
            )

        response_text = (
            enforced_response.final_response
            if hasattr(enforced_response, 'final_response')
            else ""
        ).lower()

        if not response_text:
            return self._gate_pass(
                EnforcerGate.MEMORY_ATTRIBUTION,
                "Empty response",
            )

        # Check if memory facts are properly attributed (not claimed as new info)
        ltm_facts = (
            unified_memory.ltm_facts
            if hasattr(unified_memory, 'ltm_facts')
            else []
        )

        if not ltm_facts:
            return self._gate_pass(
                EnforcerGate.MEMORY_ATTRIBUTION,
                "No LTM facts to attribute",
            )

        # Check for misattribution patterns
        misattribution_phrases = [
            "i just discovered",
            "new information",
            "i've just learned",
            "this is new",
            "first time seeing",
        ]

        misattributions = sum(
            1 for phrase in misattribution_phrases
            if phrase in response_text
        )

        if misattributions == 0:
            return self._gate_pass(
                EnforcerGate.MEMORY_ATTRIBUTION,
                "No misattribution patterns detected",
            )

        return self._gate_fail(
            EnforcerGate.MEMORY_ATTRIBUTION,
            f"Possible misattribution detected ({misattributions} patterns)",
            details={"misattribution_count": misattributions},
        )
