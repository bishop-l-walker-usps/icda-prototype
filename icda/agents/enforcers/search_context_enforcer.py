"""Search Context Enforcer - Validates memory context for search operations.

Ensures that memory context enhances (not corrupts) search operations
and the SearchAgent receives valid, relevant context.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class SearchContextEnforcer(BaseEnforcer):
    """Enforcer for search context quality.

    Quality Gates:
    - CONTEXT_RELEVANCE: Memory context relevant to query
    - FILTER_PRESERVATION: Memory doesn't override explicit filters
    - PRONOUN_RESOLUTION_ACCURACY: Pronouns resolve correctly
    - SEARCH_CONFIDENCE_STABLE: Search confidence not degraded
    """

    # Thresholds
    RELEVANCE_THRESHOLD = 0.7      # 70% relevance
    PRONOUN_ACCURACY_THRESHOLD = 0.95  # 95% accuracy
    CONFIDENCE_DELTA_THRESHOLD = -0.1  # Max allowed confidence drop

    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        """Initialize SearchContextEnforcer."""
        super().__init__(
            name="SearchContextEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.CONTEXT_RELEVANCE,
            EnforcerGate.FILTER_PRESERVATION,
            EnforcerGate.PRONOUN_RESOLUTION_ACCURACY,
            EnforcerGate.SEARCH_CONFIDENCE_STABLE,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run search context gates.

        Args:
            context: Must contain:
                - unified_memory: UnifiedMemoryContext
                - query: Original query
                - parsed_query: ParsedQuery from ParserAgent
                - search_result: SearchResult (optional, for confidence check)
                - baseline_confidence: Confidence without memory (optional)

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

        # Gate 1: Context Relevance
        relevance_result = self._check_context_relevance(context)
        (gates_passed if relevance_result.passed else gates_failed).append(
            relevance_result
        )

        # Gate 2: Filter Preservation
        filter_result = self._check_filter_preservation(context)
        (gates_passed if filter_result.passed else gates_failed).append(filter_result)

        # Gate 3: Pronoun Resolution Accuracy
        pronoun_result = self._check_pronoun_resolution(context)
        (gates_passed if pronoun_result.passed else gates_failed).append(pronoun_result)

        # Gate 4: Search Confidence Stable
        confidence_result = self._check_search_confidence(context)
        (gates_passed if confidence_result.passed else gates_failed).append(
            confidence_result
        )

        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "gates_evaluated": len(gates_passed) + len(gates_failed),
            "pass_rate": result.pass_rate,
        }

        return result

    def _check_context_relevance(self, context: dict[str, Any]) -> GateResult:
        """Check that memory context is relevant to query."""
        query = context.get("query", "")
        unified_memory = context.get("unified_memory")

        if not unified_memory or not query:
            return self._gate_pass(
                EnforcerGate.CONTEXT_RELEVANCE,
                "No memory or query to validate relevance",
            )

        # Calculate relevance based on memory signals
        ltm_facts = unified_memory.ltm_facts if hasattr(unified_memory, 'ltm_facts') else []

        if not ltm_facts:
            return self._gate_pass(
                EnforcerGate.CONTEXT_RELEVANCE,
                "No LTM facts to check relevance",
            )

        # Check fact relevance scores
        relevant_facts = [f for f in ltm_facts if f.relevance_score >= self.RELEVANCE_THRESHOLD]
        relevance_rate = len(relevant_facts) / len(ltm_facts) if ltm_facts else 1.0

        if relevance_rate >= 0.5:  # At least half should be relevant
            return self._gate_pass(
                EnforcerGate.CONTEXT_RELEVANCE,
                f"{len(relevant_facts)}/{len(ltm_facts)} facts are relevant",
                threshold=self.RELEVANCE_THRESHOLD,
                actual_value=relevance_rate,
            )

        return self._gate_fail(
            EnforcerGate.CONTEXT_RELEVANCE,
            f"Only {len(relevant_facts)}/{len(ltm_facts)} facts are relevant",
            threshold=self.RELEVANCE_THRESHOLD,
            actual_value=relevance_rate,
        )

    def _check_filter_preservation(self, context: dict[str, Any]) -> GateResult:
        """Check that memory doesn't override explicit filters."""
        parsed_query = context.get("parsed_query")
        unified_memory = context.get("unified_memory")

        if not parsed_query:
            return self._gate_pass(
                EnforcerGate.FILTER_PRESERVATION,
                "No parsed query to validate filters",
            )

        # Get explicit filters from parsed query
        explicit_filters = parsed_query.filters if hasattr(parsed_query, 'filters') else {}

        if not explicit_filters:
            return self._gate_pass(
                EnforcerGate.FILTER_PRESERVATION,
                "No explicit filters in query",
            )

        # Check if memory preferences conflict with explicit filters
        if unified_memory:
            prefs = unified_memory.ltm_preferences if hasattr(unified_memory, 'ltm_preferences') else {}

            conflicts = []
            for key, value in explicit_filters.items():
                if key in prefs and prefs[key] != value:
                    conflicts.append(f"{key}: query={value}, memory={prefs[key]}")

            if conflicts:
                return self._gate_fail(
                    EnforcerGate.FILTER_PRESERVATION,
                    f"Memory conflicts with filters: {', '.join(conflicts)}",
                    details={"conflicts": conflicts},
                )

        return self._gate_pass(
            EnforcerGate.FILTER_PRESERVATION,
            f"All {len(explicit_filters)} explicit filters preserved",
            details={"filters": list(explicit_filters.keys())},
        )

    def _check_pronoun_resolution(self, context: dict[str, Any]) -> GateResult:
        """Check that pronouns resolve correctly."""
        unified_memory = context.get("unified_memory")

        if not unified_memory:
            return self._gate_pass(
                EnforcerGate.PRONOUN_RESOLUTION_ACCURACY,
                "No memory context for pronoun resolution",
            )

        resolved = unified_memory.resolved_pronouns if hasattr(unified_memory, 'resolved_pronouns') else {}

        if not resolved:
            return self._gate_pass(
                EnforcerGate.PRONOUN_RESOLUTION_ACCURACY,
                "No pronouns to resolve",
            )

        # Check that resolved pronouns map to valid entities
        local_entities = (
            unified_memory.local_context.recalled_entities
            if hasattr(unified_memory, 'local_context') and unified_memory.local_context
            else []
        )
        entity_ids = {e.entity_id for e in local_entities if hasattr(e, 'entity_id')}

        valid_resolutions = 0
        for pronoun, entity_id in resolved.items():
            if entity_id in entity_ids or entity_id.startswith("location:"):
                valid_resolutions += 1

        accuracy = valid_resolutions / len(resolved) if resolved else 1.0

        if accuracy >= self.PRONOUN_ACCURACY_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.PRONOUN_RESOLUTION_ACCURACY,
                f"Pronoun resolution accuracy {accuracy:.1%}",
                threshold=self.PRONOUN_ACCURACY_THRESHOLD,
                actual_value=accuracy,
            )

        return self._gate_fail(
            EnforcerGate.PRONOUN_RESOLUTION_ACCURACY,
            f"Pronoun resolution accuracy {accuracy:.1%} below threshold",
            threshold=self.PRONOUN_ACCURACY_THRESHOLD,
            actual_value=accuracy,
        )

    def _check_search_confidence(self, context: dict[str, Any]) -> GateResult:
        """Check that search confidence is not degraded by memory."""
        search_result = context.get("search_result")
        baseline_confidence = context.get("baseline_confidence")

        if not search_result:
            return self._gate_pass(
                EnforcerGate.SEARCH_CONFIDENCE_STABLE,
                "No search result to validate",
            )

        current_confidence = (
            search_result.search_confidence
            if hasattr(search_result, 'search_confidence')
            else 0.5
        )

        if baseline_confidence is None:
            return self._gate_pass(
                EnforcerGate.SEARCH_CONFIDENCE_STABLE,
                f"Search confidence: {current_confidence:.2f} (no baseline)",
                actual_value=current_confidence,
            )

        delta = current_confidence - baseline_confidence

        if delta >= self.CONFIDENCE_DELTA_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.SEARCH_CONFIDENCE_STABLE,
                f"Search confidence delta: {delta:+.2f}",
                threshold=self.CONFIDENCE_DELTA_THRESHOLD,
                actual_value=delta,
            )

        return self._gate_fail(
            EnforcerGate.SEARCH_CONFIDENCE_STABLE,
            f"Search confidence dropped by {-delta:.2f}",
            threshold=self.CONFIDENCE_DELTA_THRESHOLD,
            actual_value=delta,
        )
