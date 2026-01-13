"""Enforcer Coordinator - Orchestrates all memory and RAG enforcers.

Runs all 7 enforcers in sequence and aggregates results
to provide a single go/no-go decision for memory and RAG operations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base_enforcer import EnforcerResult
from .memory_integrity_enforcer import MemoryIntegrityEnforcer
from .search_context_enforcer import SearchContextEnforcer
from .nova_context_enforcer import NovaContextEnforcer
from .response_quality_enforcer import ResponseQualityEnforcer
from .functionality_preservation_enforcer import FunctionalityPreservationEnforcer
from .rag_context_enforcer import RAGContextEnforcer
from .directory_coverage_enforcer import DirectoryCoverageEnforcer

logger = logging.getLogger(__name__)


class EnforcerCoordinator:
    """Orchestrates all 7 memory and RAG enforcers in sequence.

    Execution Order:
    1. MemoryIntegrityEnforcer - Validate memory read/write
    2. SearchContextEnforcer - Validate context for search
    3. NovaContextEnforcer - Validate context for Nova
    4. ResponseQualityEnforcer - Final response validation
    5. RAGContextEnforcer - Validate knowledge chunks in context
    6. DirectoryCoverageEnforcer - Validate directory scanning
    7. FunctionalityPreservationEnforcer - Meta-validation & metrics

    The coordinator:
    - Runs enforcers in sequence (early exit on critical failure)
    - Aggregates results from all enforcers
    - Provides final pass/fail decision
    - Tracks metrics and recommendations
    """

    __slots__ = (
        "_memory_enforcer",
        "_search_enforcer",
        "_nova_enforcer",
        "_response_enforcer",
        "_rag_enforcer",
        "_directory_enforcer",
        "_preservation_enforcer",
        "_enabled",
        "_fail_fast",
    )

    def __init__(
        self,
        enabled: bool = True,
        fail_fast: bool = False,
        strict_mode: bool = False,
    ):
        """Initialize EnforcerCoordinator.

        Args:
            enabled: Whether enforcement is active.
            fail_fast: If True, stop on first critical failure.
            strict_mode: If True, enforcers use strict mode.
        """
        self._enabled = enabled
        self._fail_fast = fail_fast

        # Initialize all enforcers
        self._memory_enforcer = MemoryIntegrityEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        self._search_enforcer = SearchContextEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        self._nova_enforcer = NovaContextEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        self._response_enforcer = ResponseQualityEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        # RAG enforcers
        self._rag_enforcer = RAGContextEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        self._directory_enforcer = DirectoryCoverageEnforcer(
            enabled=enabled, strict_mode=strict_mode
        )
        self._preservation_enforcer = FunctionalityPreservationEnforcer(
            enabled=enabled, strict_mode=True  # Always strict for preservation
        )

    @property
    def enabled(self) -> bool:
        """Check if coordinator is enabled."""
        return self._enabled

    @property
    def preservation_enforcer(self) -> FunctionalityPreservationEnforcer:
        """Get the functionality preservation enforcer for direct access."""
        return self._preservation_enforcer

    async def enforce(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run all enforcers and aggregate results.

        Args:
            context: Context dict with all data needed by enforcers:
                - unified_memory: UnifiedMemoryContext
                - session_id: Session identifier
                - actor_id: Actor identifier
                - query: Original query
                - parsed_query: ParsedQuery
                - search_result: SearchResult
                - nova_response: NovaResponse
                - enforced_response: EnforcedResponse from existing EnforcerAgent
                - baseline_quality: Quality without memory (optional)
                - memory_latency_ms: Latency from memory operations

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

        # 1. Memory Integrity
        memory_result = await self._memory_enforcer.enforce(context)
        results.append(memory_result)
        if not memory_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 2. Search Context
        search_result = await self._search_enforcer.enforce(context)
        results.append(search_result)
        if not search_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 3. Nova Context
        nova_result = await self._nova_enforcer.enforce(context)
        results.append(nova_result)
        if not nova_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 4. Response Quality
        response_result = await self._response_enforcer.enforce(context)
        results.append(response_result)
        if not response_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 5. RAG Context
        rag_result = await self._rag_enforcer.enforce(context)
        results.append(rag_result)
        if not rag_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 6. Directory Coverage
        directory_result = await self._directory_enforcer.enforce(context)
        results.append(directory_result)
        if not directory_result.passed:
            all_passed = False
            if self._fail_fast:
                return self._create_response(results, start_time, all_passed)

        # 7. Functionality Preservation (final, with other results)
        preservation_context = {
            **context,
            "enforcer_results": results,
            "quality_with_memory": self._calculate_aggregate_quality(results),
        }
        preservation_result = await self._preservation_enforcer.enforce(
            preservation_context
        )
        results.append(preservation_result)
        if not preservation_result.passed:
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

    def _calculate_aggregate_quality(self, results: list[EnforcerResult]) -> float:
        """Calculate aggregate quality from enforcer results."""
        if not results:
            return 1.0

        # Weight each enforcer equally
        scores = [r.quality_score for r in results]
        return sum(scores) / len(scores)

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        stats = {
            "enabled": self._enabled,
            "fail_fast": self._fail_fast,
            "circuit_breaker_active": self._preservation_enforcer.circuit_breaker_active,
        }

        # Add quality stats from preservation enforcer
        quality_stats = self._preservation_enforcer.get_quality_stats()
        stats["quality_comparison"] = quality_stats

        return stats

    def enable_shadow_mode(self) -> None:
        """Enable shadow mode for A/B testing."""
        self._preservation_enforcer.enable_shadow_mode()

    def disable_shadow_mode(self) -> None:
        """Disable shadow mode."""
        self._preservation_enforcer.disable_shadow_mode()
