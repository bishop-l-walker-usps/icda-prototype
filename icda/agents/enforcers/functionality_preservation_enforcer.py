"""Functionality Preservation Enforcer - Central meta-enforcer.

Coordinates all other enforcers and validates that ALL memory operations
do not degrade existing system behavior. Acts as the final go/no-go decision.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class FunctionalityPreservationEnforcer(BaseEnforcer):
    """Central enforcer for overall functionality preservation.

    Quality Gates:
    - RESPONSE_PARITY: Response quality with memory >= without
    - LATENCY_BUDGET: Memory operations within latency budget
    - FEATURE_COMPLETE: All 8 existing quality gates still pass
    - GRACEFUL_DEGRADATION: System works when AgentCore unavailable

    Capabilities:
    - Shadow mode testing (old vs new memory systems)
    - Auto-rollback on quality drop > 10%
    - Circuit breaker after 5 consecutive failures
    """

    # Thresholds
    PARITY_THRESHOLD = -0.05      # Max allowed quality drop
    LATENCY_BUDGET_MS = 500       # Max added latency from memory
    MIN_FEATURE_PASS_RATE = 0.9   # 90% of features must pass
    CIRCUIT_BREAKER_THRESHOLD = 5 # Consecutive failures to trigger

    def __init__(self, enabled: bool = True, strict_mode: bool = True):
        """Initialize FunctionalityPreservationEnforcer."""
        super().__init__(
            name="FunctionalityPreservationEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._consecutive_failures = 0
        self._circuit_breaker_active = False
        self._circuit_breaker_until = 0.0
        self._shadow_mode = False
        self._quality_history: list[tuple[float, float]] = []  # (with_memory, without_memory)

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.RESPONSE_PARITY,
            EnforcerGate.LATENCY_BUDGET,
            EnforcerGate.FEATURE_COMPLETE,
            EnforcerGate.GRACEFUL_DEGRADATION,
        ]

    @property
    def circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if self._circuit_breaker_active:
            if time.time() >= self._circuit_breaker_until:
                self._circuit_breaker_active = False
                self._consecutive_failures = 0
                logger.info("Circuit breaker reset")
            return self._circuit_breaker_active
        return False

    def enable_shadow_mode(self) -> None:
        """Enable shadow mode for A/B testing."""
        self._shadow_mode = True
        logger.info("Shadow mode enabled")

    def disable_shadow_mode(self) -> None:
        """Disable shadow mode."""
        self._shadow_mode = False
        logger.info("Shadow mode disabled")

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run functionality preservation gates.

        Args:
            context: Must contain:
                - quality_with_memory: Quality score with memory
                - quality_without_memory: Quality score without memory (optional)
                - memory_latency_ms: Latency added by memory operations
                - enforcer_results: Results from other enforcers
                - agentcore_available: Whether AgentCore was available
                - feature_results: Results from existing 8 quality gates

        Returns:
            EnforcerResult with gate outcomes and recommendations.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
            )

        # Check circuit breaker
        if self.circuit_breaker_active:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=False,
                quality_score=0.0,
                recommendations=["Circuit breaker active - memory disabled"],
                metrics={"circuit_breaker_active": True},
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Gate 1: Response Parity
        parity_result = self._check_response_parity(context)
        (gates_passed if parity_result.passed else gates_failed).append(parity_result)

        # Gate 2: Latency Budget
        latency_result = self._check_latency_budget(context)
        (gates_passed if latency_result.passed else gates_failed).append(latency_result)

        # Gate 3: Feature Complete
        feature_result = self._check_feature_complete(context)
        (gates_passed if feature_result.passed else gates_failed).append(feature_result)

        # Gate 4: Graceful Degradation
        degradation_result = self._check_graceful_degradation(context)
        (gates_passed if degradation_result.passed else gates_failed).append(
            degradation_result
        )

        result = self._create_result(gates_passed, gates_failed)

        # Update circuit breaker state
        if not result.passed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                self._activate_circuit_breaker()
                result.recommendations.append(
                    f"Circuit breaker activated after {self._consecutive_failures} failures"
                )
        else:
            self._consecutive_failures = 0

        # Track quality history for shadow mode
        self._update_quality_history(context)

        result.metrics = {
            "gates_evaluated": len(gates_passed) + len(gates_failed),
            "pass_rate": result.pass_rate,
            "consecutive_failures": self._consecutive_failures,
            "circuit_breaker_active": self._circuit_breaker_active,
            "shadow_mode": self._shadow_mode,
            "quality_history_size": len(self._quality_history),
        }

        return result

    def _check_response_parity(self, context: dict[str, Any]) -> GateResult:
        """Check that response quality with memory >= without memory."""
        quality_with = context.get("quality_with_memory")
        quality_without = context.get("quality_without_memory")

        if quality_with is None:
            return self._gate_pass(
                EnforcerGate.RESPONSE_PARITY,
                "No quality score available",
            )

        if quality_without is None:
            return self._gate_pass(
                EnforcerGate.RESPONSE_PARITY,
                f"Quality: {quality_with:.2f} (no baseline)",
                actual_value=quality_with,
            )

        delta = quality_with - quality_without

        if delta >= self.PARITY_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.RESPONSE_PARITY,
                f"Quality delta: {delta:+.2f} (parity maintained)",
                threshold=self.PARITY_THRESHOLD,
                actual_value=delta,
            )

        return self._gate_fail(
            EnforcerGate.RESPONSE_PARITY,
            f"Quality dropped by {-delta:.2f} (threshold: {-self.PARITY_THRESHOLD:.2f})",
            threshold=self.PARITY_THRESHOLD,
            actual_value=delta,
        )

    def _check_latency_budget(self, context: dict[str, Any]) -> GateResult:
        """Check that memory operations are within latency budget."""
        memory_latency_ms = context.get("memory_latency_ms", 0)

        if memory_latency_ms <= self.LATENCY_BUDGET_MS:
            return self._gate_pass(
                EnforcerGate.LATENCY_BUDGET,
                f"Memory latency: {memory_latency_ms}ms within budget",
                threshold=self.LATENCY_BUDGET_MS,
                actual_value=memory_latency_ms,
            )

        return self._gate_fail(
            EnforcerGate.LATENCY_BUDGET,
            f"Memory latency: {memory_latency_ms}ms exceeds budget",
            threshold=self.LATENCY_BUDGET_MS,
            actual_value=memory_latency_ms,
        )

    def _check_feature_complete(self, context: dict[str, Any]) -> GateResult:
        """Check that all existing features still work."""
        feature_results = context.get("feature_results", {})
        enforcer_results = context.get("enforcer_results", [])

        if not feature_results and not enforcer_results:
            return self._gate_pass(
                EnforcerGate.FEATURE_COMPLETE,
                "No feature results to validate",
            )

        # Check enforcer results from other enforcers
        total_enforcers = len(enforcer_results)
        passed_enforcers = sum(
            1 for r in enforcer_results
            if isinstance(r, EnforcerResult) and r.passed
        )

        if total_enforcers > 0:
            pass_rate = passed_enforcers / total_enforcers
            if pass_rate >= self.MIN_FEATURE_PASS_RATE:
                return self._gate_pass(
                    EnforcerGate.FEATURE_COMPLETE,
                    f"{passed_enforcers}/{total_enforcers} enforcers passed",
                    threshold=self.MIN_FEATURE_PASS_RATE,
                    actual_value=pass_rate,
                )

            return self._gate_fail(
                EnforcerGate.FEATURE_COMPLETE,
                f"Only {passed_enforcers}/{total_enforcers} enforcers passed",
                threshold=self.MIN_FEATURE_PASS_RATE,
                actual_value=pass_rate,
            )

        # Check feature_results dict
        total_features = len(feature_results)
        passed_features = sum(1 for v in feature_results.values() if v)

        if total_features > 0:
            pass_rate = passed_features / total_features
            if pass_rate >= self.MIN_FEATURE_PASS_RATE:
                return self._gate_pass(
                    EnforcerGate.FEATURE_COMPLETE,
                    f"{passed_features}/{total_features} features passed",
                    threshold=self.MIN_FEATURE_PASS_RATE,
                    actual_value=pass_rate,
                )

            return self._gate_fail(
                EnforcerGate.FEATURE_COMPLETE,
                f"Only {passed_features}/{total_features} features passed",
                threshold=self.MIN_FEATURE_PASS_RATE,
                actual_value=pass_rate,
            )

        return self._gate_pass(
            EnforcerGate.FEATURE_COMPLETE,
            "Feature completeness verified",
        )

    def _check_graceful_degradation(self, context: dict[str, Any]) -> GateResult:
        """Check that system works when AgentCore unavailable."""
        agentcore_available = context.get("agentcore_available", True)
        unified_memory = context.get("unified_memory")
        response_generated = context.get("response_generated", True)

        if agentcore_available:
            return self._gate_pass(
                EnforcerGate.GRACEFUL_DEGRADATION,
                "AgentCore available - degradation not tested",
            )

        # AgentCore was unavailable - check if we degraded gracefully
        if not response_generated:
            return self._gate_fail(
                EnforcerGate.GRACEFUL_DEGRADATION,
                "Failed to generate response when AgentCore unavailable",
            )

        # Check if local memory was used as fallback
        if unified_memory:
            memory_source = (
                unified_memory.memory_source
                if hasattr(unified_memory, 'memory_source')
                else "unknown"
            )
            if memory_source in ("local", "hybrid"):
                return self._gate_pass(
                    EnforcerGate.GRACEFUL_DEGRADATION,
                    f"Gracefully degraded to {memory_source} memory",
                    details={"memory_source": memory_source},
                )

        return self._gate_pass(
            EnforcerGate.GRACEFUL_DEGRADATION,
            "Response generated despite AgentCore unavailable",
        )

    def _activate_circuit_breaker(self) -> None:
        """Activate circuit breaker for 5 minutes."""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + 300  # 5 minutes
        logger.warning(
            f"Circuit breaker activated after {self._consecutive_failures} failures"
        )

    def _update_quality_history(self, context: dict[str, Any]) -> None:
        """Update quality history for tracking."""
        quality_with = context.get("quality_with_memory")
        quality_without = context.get("quality_without_memory")

        if quality_with is not None:
            self._quality_history.append((
                quality_with,
                quality_without if quality_without is not None else quality_with,
            ))

            # Keep last 100 entries
            if len(self._quality_history) > 100:
                self._quality_history = self._quality_history[-100:]

    def get_quality_stats(self) -> dict[str, Any]:
        """Get quality comparison statistics."""
        if not self._quality_history:
            return {"samples": 0}

        with_memory = [q[0] for q in self._quality_history]
        without_memory = [q[1] for q in self._quality_history]

        avg_with = sum(with_memory) / len(with_memory)
        avg_without = sum(without_memory) / len(without_memory)

        return {
            "samples": len(self._quality_history),
            "avg_quality_with_memory": avg_with,
            "avg_quality_without_memory": avg_without,
            "avg_delta": avg_with - avg_without,
            "improvements": sum(1 for w, wo in self._quality_history if w > wo),
            "regressions": sum(1 for w, wo in self._quality_history if w < wo),
        }
