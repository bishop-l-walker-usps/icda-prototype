"""Address Enforcer Coordinator - Lightweight single-address coordination.

Coordinates all 4 address validation enforcers for single-address
processing. For batch processing, use BatchOrchestratorAgent instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..base_enforcer import EnforcerResult
from .completion_enforcer import CompletionEnforcerAgent
from .correction_enforcer import CorrectionEnforcerAgent
from .match_confidence_enforcer import MatchConfidenceEnforcerAgent
from .models import AddressEnforcerGate
from .normalization_enforcer import NormalizationEnforcerAgent

logger = logging.getLogger(__name__)


class AddressEnforcerCoordinator:
    """Lightweight coordinator for single-address enforcement.

    Runs all 4 address enforcers in sequence and aggregates results.
    For batch processing with concurrent execution, use BatchOrchestratorAgent.
    """

    __slots__ = (
        "_normalization_enforcer",
        "_completion_enforcer",
        "_correction_enforcer",
        "_match_confidence_enforcer",
        "_enabled",
        "_fail_fast",
    )

    def __init__(
        self,
        enabled: bool = True,
        fail_fast: bool = False,
        strict_mode: bool = False,
    ):
        """Initialize AddressEnforcerCoordinator.

        Args:
            enabled: Whether enforcement is active.
            fail_fast: If True, stop on first critical failure.
            strict_mode: If True, enforcers use strict mode.
        """
        self._enabled = enabled
        self._fail_fast = fail_fast

        # Initialize all enforcers
        self._normalization_enforcer = NormalizationEnforcerAgent(
            enabled=enabled, strict_mode=strict_mode
        )
        self._completion_enforcer = CompletionEnforcerAgent(
            enabled=enabled, strict_mode=strict_mode
        )
        self._correction_enforcer = CorrectionEnforcerAgent(
            enabled=enabled, strict_mode=strict_mode
        )
        self._match_confidence_enforcer = MatchConfidenceEnforcerAgent(
            enabled=enabled, strict_mode=strict_mode
        )

    @property
    def enabled(self) -> bool:
        """Check if coordinator is enabled."""
        return self._enabled

    async def enforce(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run all address enforcers and aggregate results.

        Args:
            context: Context dict with all data needed by enforcers:
                - original_address: Original input address
                - normalized_address: Normalized/standardized address
                - validation_result: Validation result object
                - original_components: Original parsed components
                - completed_components: Components after completion
                - corrected_components: Components after correction
                - error_type: Expected error type
                - expected_correction: Expected correction details
                - overall_confidence: Confidence score

        Returns:
            Dict with:
                - passed: Overall pass/fail
                - quality_score: Aggregate quality score
                - results: List of EnforcerResult from each enforcer
                - recommendations: Combined recommendations
                - metrics: Combined metrics
        """
        if not self._enabled:
            return {
                "passed": True,
                "quality_score": 1.0,
                "results": [],
                "recommendations": [],
                "metrics": {"enforcement_skipped": True},
            }

        start_time = time.time()
        results: list[EnforcerResult] = []
        all_passed = True

        # 1. Normalization Enforcer
        norm_result = await self._normalization_enforcer.enforce(context)
        results.append(norm_result)
        if not norm_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 2. Completion Enforcer
        comp_result = await self._completion_enforcer.enforce(context)
        results.append(comp_result)
        if not comp_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 3. Correction Enforcer
        corr_result = await self._correction_enforcer.enforce(context)
        results.append(corr_result)
        if not corr_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 4. Match Confidence Enforcer
        match_result = await self._match_confidence_enforcer.enforce(context)
        results.append(match_result)
        if not match_result.passed:
            all_passed = False

        return self._create_response(results, start_time, all_passed)

    def _create_response(
        self,
        results: list[EnforcerResult],
        start_time: float,
        all_passed: bool,
    ) -> dict[str, Any]:
        """Create aggregated response from enforcer results."""
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Aggregate quality score
        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0

        # Combine recommendations
        all_recommendations = []
        for r in results:
            all_recommendations.extend(r.recommendations)

        # Combine metrics
        combined_metrics = {
            "enforcement_time_ms": elapsed_ms,
            "enforcers_run": len(results),
            "enforcers_passed": sum(1 for r in results if r.passed),
            "total_gates_passed": sum(len(r.gates_passed) for r in results),
            "total_gates_failed": sum(len(r.gates_failed) for r in results),
        }

        # Add individual enforcer metrics
        for r in results:
            combined_metrics[f"{r.enforcer_name}_passed"] = r.passed
            combined_metrics[f"{r.enforcer_name}_quality"] = r.quality_score

        return {
            "passed": all_passed,
            "quality_score": avg_quality,
            "results": results,
            "recommendations": all_recommendations,
            "metrics": combined_metrics,
        }

    def get_gates(self) -> list[AddressEnforcerGate]:
        """Get all gates checked by this coordinator."""
        gates = []
        gates.extend(self._normalization_enforcer.get_gates())
        gates.extend(self._completion_enforcer.get_gates())
        gates.extend(self._correction_enforcer.get_gates())
        gates.extend(self._match_confidence_enforcer.get_gates())
        return gates

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "enabled": self._enabled,
            "fail_fast": self._fail_fast,
            "total_gates": len(self.get_gates()),
            "enforcers": [
                "NormalizationEnforcer",
                "CompletionEnforcer",
                "CorrectionEnforcer",
                "MatchConfidenceEnforcer",
            ],
        }
