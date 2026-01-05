"""Puerto Rico Address Preservation Enforcer.

Ensures PR address handling quality is maintained across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerGate, EnforcerResult, GateResult


@dataclass(slots=True)
class PRAddressMetrics:
    """Metrics for PR address quality tracking."""

    total_pr_addresses: int = 0
    addresses_with_urbanization: int = 0
    urbanization_detection_rate: float = 0.0
    avg_confidence: float = 0.0


class PRAddressPreservationEnforcer(BaseEnforcer):
    """Enforcer for Puerto Rico address quality preservation.

    Ensures:
    - Urbanization detection rate stays above threshold
    - PR address format is preserved
    - Confidence scores don't regress
    """

    MIN_URB_DETECTION_RATE: float = 0.85  # Must detect URB in 85% of PR addresses
    MAX_CONFIDENCE_DROP: float = 0.10  # Max allowed confidence regression

    def __init__(self, enabled: bool = True, strict_mode: bool = True):
        super().__init__(
            name="PRAddressPreservationEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._baseline_metrics: dict[str, float] = {}
        self._pr_quality_history: list[PRAddressMetrics] = []

    def get_gates(self) -> list[EnforcerGate]:
        """Return the gates this enforcer checks."""
        return [
            EnforcerGate.PR_URBANIZATION_REQUIRED,
            EnforcerGate.PR_FORMAT_PRESERVED,
            EnforcerGate.PR_CONFIDENCE_MAINTAINED,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run PR address preservation gates.

        Args:
            context: Pipeline context with addresses and results.

        Returns:
            EnforcerResult with gate outcomes.
        """
        if not self.enabled:
            return EnforcerResult(
                enforcer_name=self.name,
                passed=True,
                gates_passed=[],
                gates_failed=[],
                quality_score=1.0,
                recommendations=["PR Address Enforcer disabled"],
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Extract PR addresses from context
        addresses = context.get("addresses", [])
        pr_addresses = [a for a in addresses if self._is_pr_address(a)]

        if not pr_addresses:
            return EnforcerResult(
                enforcer_name=self.name,
                passed=True,
                gates_passed=[],
                gates_failed=[],
                quality_score=1.0,
                recommendations=["No PR addresses to validate"],
            )

        # Calculate metrics
        metrics = self._calculate_metrics(pr_addresses)

        # Gate 1: Urbanization detection rate
        urb_gate = self._check_urbanization_rate(metrics)
        if urb_gate.passed:
            gates_passed.append(urb_gate)
        else:
            gates_failed.append(urb_gate)

        # Gate 2: Format preserved
        format_gate = self._check_format_preserved(pr_addresses)
        if format_gate.passed:
            gates_passed.append(format_gate)
        else:
            gates_failed.append(format_gate)

        # Gate 3: Confidence maintained
        conf_gate = self._check_confidence_maintained(metrics)
        if conf_gate.passed:
            gates_passed.append(conf_gate)
        else:
            gates_failed.append(conf_gate)

        # Calculate quality score
        total_gates = len(gates_passed) + len(gates_failed)
        quality_score = len(gates_passed) / total_gates if total_gates > 0 else 1.0

        # Store metrics for history
        self._pr_quality_history.append(metrics)

        # Build recommendations from failed gates
        recommendations = [
            f"Fix {g.gate.value}: {g.message}" for g in gates_failed
        ]
        if not recommendations:
            recommendations = [f"PR validation: {len(gates_passed)}/{total_gates} gates passed"]

        return EnforcerResult(
            enforcer_name=self.name,
            passed=len(gates_failed) == 0,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            quality_score=quality_score,
            recommendations=recommendations,
            metrics={
                "pr_addresses_checked": len(pr_addresses),
                "urbanization_rate": metrics.urbanization_detection_rate,
                "avg_confidence": metrics.avg_confidence,
            },
        )

    def _is_pr_address(self, address: dict[str, Any]) -> bool:
        """Check if address is Puerto Rico."""
        if address.get("is_puerto_rico"):
            return True
        zip_code = address.get("zip_code", address.get("zip", ""))
        if zip_code and len(str(zip_code)) >= 3:
            prefix = str(zip_code)[:3]
            return prefix in ("006", "007", "008", "009")
        return False

    def _calculate_metrics(self, pr_addresses: list[dict]) -> PRAddressMetrics:
        """Calculate PR address quality metrics."""
        total = len(pr_addresses)
        with_urb = sum(1 for a in pr_addresses if a.get("urbanization"))
        confidences = [a.get("confidence", 0.0) for a in pr_addresses]

        return PRAddressMetrics(
            total_pr_addresses=total,
            addresses_with_urbanization=with_urb,
            urbanization_detection_rate=with_urb / total if total > 0 else 0.0,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )

    def _check_urbanization_rate(self, metrics: PRAddressMetrics) -> GateResult:
        """Check urbanization detection rate gate."""
        passed = metrics.urbanization_detection_rate >= self.MIN_URB_DETECTION_RATE

        return GateResult(
            gate=EnforcerGate.PR_URBANIZATION_REQUIRED,
            passed=passed,
            message=f"URB detection rate: {metrics.urbanization_detection_rate:.1%}",
            threshold=self.MIN_URB_DETECTION_RATE,
            actual_value=metrics.urbanization_detection_rate,
        )

    def _check_format_preserved(self, pr_addresses: list[dict]) -> GateResult:
        """Check PR address format is correct."""
        # Check that PR addresses have proper format markers
        issues = []
        for addr in pr_addresses:
            if addr.get("urbanization") and not addr.get("formatted", "").upper().startswith("URB"):
                # Urbanization exists but formatted doesn't show URB
                issues.append(addr.get("zip_code", "unknown"))

        passed = len(issues) == 0

        return GateResult(
            gate=EnforcerGate.PR_FORMAT_PRESERVED,
            passed=passed,
            message=f"Format issues: {len(issues)}" if issues else "Format preserved",
            threshold=0,
            actual_value=len(issues),
        )

    def _check_confidence_maintained(self, metrics: PRAddressMetrics) -> GateResult:
        """Check confidence hasn't dropped below baseline."""
        baseline = self._baseline_metrics.get("avg_confidence", 0.0)
        if baseline == 0.0:
            # No baseline yet, pass
            return GateResult(
                gate=EnforcerGate.PR_CONFIDENCE_MAINTAINED,
                passed=True,
                message="No baseline established",
                threshold=self.MAX_CONFIDENCE_DROP,
                actual_value=0.0,
            )

        drop = baseline - metrics.avg_confidence
        passed = drop <= self.MAX_CONFIDENCE_DROP

        return GateResult(
            gate=EnforcerGate.PR_CONFIDENCE_MAINTAINED,
            passed=passed,
            message=f"Confidence drop: {drop:.1%}",
            threshold=self.MAX_CONFIDENCE_DROP,
            actual_value=drop,
        )

    def set_baseline(self, metrics: PRAddressMetrics) -> None:
        """Set baseline metrics for regression detection."""
        self._baseline_metrics = {
            "avg_confidence": metrics.avg_confidence,
            "urbanization_rate": metrics.urbanization_detection_rate,
        }

    def get_quality_history(self) -> list[PRAddressMetrics]:
        """Get the history of PR address quality metrics."""
        return self._pr_quality_history.copy()

    def clear_history(self) -> None:
        """Clear the quality history."""
        self._pr_quality_history.clear()
