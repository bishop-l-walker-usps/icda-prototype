"""Match Confidence Enforcer Agent - Address matching quality gates.

Agent 4 of 6 in the address validation enforcer pipeline.
Validates match confidence and ambiguity resolution.

Quality Gates:
1. CONFIDENCE_ABOVE_THRESHOLD - Main match >= 0.70
2. ALTERNATIVES_SCORED - Alternatives have scores
3. NO_AMBIGUITY - Clear winner (gap >= 15% from #2)
4. MATCH_TYPE_APPROPRIATE - Match type fits confidence level
5. MULTI_STATE_RESOLVED - State ambiguity resolved
6. COMPONENT_ALIGNMENT - Components align with match
"""

from __future__ import annotations

from typing import Any

from ..base_enforcer import BaseEnforcer, EnforcerResult, GateResult
from .models import AddressEnforcerGate, MatchConfidenceMetrics


# Cities that exist in multiple states
MULTI_STATE_CITIES = {
    "SPRINGFIELD": ["IL", "MA", "MO", "OH", "OR"],
    "COLUMBUS": ["GA", "IN", "MS", "NE", "OH"],
    "WASHINGTON": ["DC", "GA", "IN", "NC", "PA", "UT"],
    "CLINTON": ["CT", "IA", "MA", "MD", "MI", "MS", "NC", "SC", "TN"],
    "FRANKLIN": ["IN", "KY", "LA", "MA", "NC", "NH", "NJ", "OH", "PA", "TN", "VA", "WI"],
    "MADISON": ["AL", "CT", "FL", "GA", "IN", "MS", "NE", "NJ", "OH", "SD", "TN", "VA", "WI"],
    "JACKSON": ["AL", "CA", "GA", "KY", "LA", "MI", "MN", "MS", "MO", "NC", "OH", "TN", "WY"],
    "PORTLAND": ["ME", "OR"],
    "AURORA": ["CO", "IL", "NE", "OH"],
    "RICHMOND": ["CA", "IN", "KY", "TX", "VA"],
}


class MatchConfidenceEnforcerAgent(BaseEnforcer):
    """Enforcer agent for address match confidence validation.

    Validates that address matching produced high-confidence results
    with clear winners and resolved ambiguities.
    """

    __slots__ = ("_confidence_threshold", "_ambiguity_gap_threshold")

    def __init__(
        self,
        enabled: bool = True,
        strict_mode: bool = False,
        confidence_threshold: float = 0.70,
        ambiguity_gap_threshold: float = 0.15,
    ):
        """Initialize MatchConfidenceEnforcerAgent.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the entire check.
            confidence_threshold: Minimum confidence for primary match.
            ambiguity_gap_threshold: Minimum gap between #1 and #2 matches.
        """
        super().__init__(
            name="MatchConfidenceEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._confidence_threshold = confidence_threshold
        self._ambiguity_gap_threshold = ambiguity_gap_threshold

    def get_gates(self) -> list[AddressEnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            AddressEnforcerGate.CONFIDENCE_ABOVE_THRESHOLD,
            AddressEnforcerGate.ALTERNATIVES_SCORED,
            AddressEnforcerGate.NO_AMBIGUITY,
            AddressEnforcerGate.MATCH_TYPE_APPROPRIATE,
            AddressEnforcerGate.MULTI_STATE_RESOLVED,
            AddressEnforcerGate.COMPONENT_ALIGNMENT,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run match confidence quality gates.

        Args:
            context: Dictionary containing:
                - validation_result: Validation result object
                - alternatives: List of alternative matches
                - match_type: Type of match (exact, fuzzy, partial)
                - components: Parsed address components
                - overall_confidence: Primary match confidence

        Returns:
            EnforcerResult with gate results and match confidence metrics.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
                metrics={"enforcement_skipped": True},
            )

        # Extract data from context
        validation_result = context.get("validation_result")
        alternatives = context.get("alternatives", [])
        match_type = context.get("match_type", "")
        components = context.get("components", {})

        # Get confidence from validation result
        primary_confidence = context.get("overall_confidence", 0.0)
        if validation_result and hasattr(validation_result, "overall_confidence"):
            primary_confidence = validation_result.overall_confidence

        metrics = MatchConfidenceMetrics(
            primary_confidence=primary_confidence,
            match_type=match_type,
        )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Analyze alternatives
        self._analyze_alternatives(alternatives, metrics)

        # Analyze components
        self._analyze_components(components, metrics)

        # Gate 1: Confidence Above Threshold
        conf_result = self._check_confidence_threshold(metrics)
        (gates_passed if conf_result.passed else gates_failed).append(conf_result)

        # Gate 2: Alternatives Scored
        alt_result = self._check_alternatives_scored(alternatives, metrics)
        (gates_passed if alt_result.passed else gates_failed).append(alt_result)

        # Gate 3: No Ambiguity
        ambig_result = self._check_no_ambiguity(metrics)
        (gates_passed if ambig_result.passed else gates_failed).append(ambig_result)

        # Gate 4: Match Type Appropriate
        type_result = self._check_match_type(validation_result, metrics)
        (gates_passed if type_result.passed else gates_failed).append(type_result)

        # Gate 5: Multi-State Resolved
        state_result = self._check_multi_state_resolved(components, metrics)
        (gates_passed if state_result.passed else gates_failed).append(state_result)

        # Gate 6: Component Alignment
        align_result = self._check_component_alignment(validation_result, components, metrics)
        (gates_passed if align_result.passed else gates_failed).append(align_result)

        # Create result
        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "match_confidence": metrics.to_dict(),
            "is_ambiguous": metrics.is_ambiguous,
        }

        return result

    def _analyze_alternatives(
        self,
        alternatives: list[Any],
        metrics: MatchConfidenceMetrics,
    ) -> None:
        """Analyze alternative matches."""
        metrics.alternatives_count = len(alternatives)

        if alternatives:
            # Get second best confidence
            alt_confidences = []
            for alt in alternatives:
                if isinstance(alt, dict):
                    conf = alt.get("confidence", alt.get("score", 0.0))
                elif hasattr(alt, "confidence"):
                    conf = alt.confidence
                elif hasattr(alt, "score"):
                    conf = alt.score
                else:
                    conf = 0.0
                alt_confidences.append(conf)

            if alt_confidences:
                metrics.secondary_confidence = max(alt_confidences)

        # Calculate gap
        metrics.confidence_gap = metrics.primary_confidence - metrics.secondary_confidence

        # Calculate ambiguity score (higher = more ambiguous)
        if metrics.secondary_confidence > 0:
            metrics.ambiguity_score = metrics.secondary_confidence / max(
                metrics.primary_confidence, 0.01
            )
        else:
            metrics.ambiguity_score = 0.0

    def _analyze_components(
        self,
        components: dict[str, Any],
        metrics: MatchConfidenceMetrics,
    ) -> None:
        """Analyze component-level match quality."""
        city = components.get("city", "").upper()
        state = components.get("state", "").upper()

        # Check for multi-state city
        if city in MULTI_STATE_CITIES:
            possible_states = MULTI_STATE_CITIES[city]
            metrics.multi_state_candidates = possible_states

            if state and state in possible_states:
                # State is specified and valid
                pass
            elif not state:
                # State not specified for multi-state city
                metrics.ambiguity_score = max(metrics.ambiguity_score, 0.5)

    def _check_confidence_threshold(
        self,
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if primary confidence meets threshold."""
        if metrics.primary_confidence >= self._confidence_threshold:
            return self._gate_pass(
                gate=AddressEnforcerGate.CONFIDENCE_ABOVE_THRESHOLD,
                message=f"Match confidence {metrics.primary_confidence:.1%} meets threshold",
                threshold=self._confidence_threshold,
                actual_value=metrics.primary_confidence,
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.CONFIDENCE_ABOVE_THRESHOLD,
                message=f"Match confidence {metrics.primary_confidence:.1%} below threshold",
                threshold=self._confidence_threshold,
                actual_value=metrics.primary_confidence,
            )

    def _check_alternatives_scored(
        self,
        alternatives: list[Any],
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if alternatives were properly scored."""
        if not alternatives:
            # No alternatives is acceptable
            return self._gate_pass(
                gate=AddressEnforcerGate.ALTERNATIVES_SCORED,
                message="No alternatives to score",
                details={"alternatives_count": 0},
            )

        # Check if alternatives have scores
        scored_count = 0
        for alt in alternatives:
            has_score = False
            if isinstance(alt, dict):
                has_score = "confidence" in alt or "score" in alt
            elif hasattr(alt, "confidence") or hasattr(alt, "score"):
                has_score = True

            if has_score:
                scored_count += 1

        if scored_count == len(alternatives):
            return self._gate_pass(
                gate=AddressEnforcerGate.ALTERNATIVES_SCORED,
                message=f"All {scored_count} alternatives properly scored",
                details={"scored": scored_count, "total": len(alternatives)},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.ALTERNATIVES_SCORED,
                message=f"Only {scored_count}/{len(alternatives)} alternatives scored",
                details={"scored": scored_count, "total": len(alternatives)},
            )

    def _check_no_ambiguity(
        self,
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if there's a clear winner without ambiguity."""
        if metrics.secondary_confidence == 0:
            # No alternatives = no ambiguity
            return self._gate_pass(
                gate=AddressEnforcerGate.NO_AMBIGUITY,
                message="No alternatives, clear match",
                details={"confidence_gap": 1.0},
            )

        if metrics.confidence_gap >= self._ambiguity_gap_threshold:
            return self._gate_pass(
                gate=AddressEnforcerGate.NO_AMBIGUITY,
                message=f"Clear winner with {metrics.confidence_gap:.1%} gap",
                threshold=self._ambiguity_gap_threshold,
                actual_value=metrics.confidence_gap,
                details={
                    "primary": metrics.primary_confidence,
                    "secondary": metrics.secondary_confidence,
                },
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.NO_AMBIGUITY,
                message=f"Ambiguous match: gap only {metrics.confidence_gap:.1%}",
                threshold=self._ambiguity_gap_threshold,
                actual_value=metrics.confidence_gap,
                details={
                    "primary": metrics.primary_confidence,
                    "secondary": metrics.secondary_confidence,
                    "ambiguity_score": metrics.ambiguity_score,
                },
            )

    def _check_match_type(
        self,
        validation_result: Any,
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if match type is appropriate for confidence level."""
        # Determine match type
        match_type = metrics.match_type
        if validation_result:
            if hasattr(validation_result, "status"):
                status = str(validation_result.status.value if hasattr(
                    validation_result.status, "value"
                ) else validation_result.status)
                match_type = status.lower()
            elif hasattr(validation_result, "match_type"):
                match_type = validation_result.match_type

        metrics.match_type = match_type

        # Define expected confidence ranges by match type
        type_thresholds = {
            "verified": 0.90,
            "valid": 0.85,
            "corrected": 0.70,
            "partial": 0.50,
            "suggested": 0.40,
            "invalid": 0.0,
        }

        # Check if confidence matches type
        expected_min = type_thresholds.get(match_type, 0.0)

        if metrics.primary_confidence >= expected_min:
            return self._gate_pass(
                gate=AddressEnforcerGate.MATCH_TYPE_APPROPRIATE,
                message=f"Match type '{match_type}' appropriate for confidence {metrics.primary_confidence:.1%}",
                details={
                    "match_type": match_type,
                    "expected_min": expected_min,
                    "actual": metrics.primary_confidence,
                },
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.MATCH_TYPE_APPROPRIATE,
                message=f"Match type '{match_type}' too high for confidence {metrics.primary_confidence:.1%}",
                details={
                    "match_type": match_type,
                    "expected_min": expected_min,
                    "actual": metrics.primary_confidence,
                },
            )

    def _check_multi_state_resolved(
        self,
        components: dict[str, Any],
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if multi-state city ambiguity is resolved."""
        city = components.get("city", "").upper()
        state = components.get("state", "").upper()

        if city not in MULTI_STATE_CITIES:
            # Not a multi-state city
            return self._gate_pass(
                gate=AddressEnforcerGate.MULTI_STATE_RESOLVED,
                message="Not a multi-state city",
                details={"city": city, "is_multi_state": False},
            )

        possible_states = MULTI_STATE_CITIES[city]
        metrics.multi_state_candidates = possible_states

        if state and state in possible_states:
            return self._gate_pass(
                gate=AddressEnforcerGate.MULTI_STATE_RESOLVED,
                message=f"Multi-state city '{city}' resolved to '{state}'",
                details={
                    "city": city,
                    "state": state,
                    "possible_states": possible_states,
                },
            )
        elif state:
            # State specified but not in expected list (could be valid)
            return self._gate_pass(
                gate=AddressEnforcerGate.MULTI_STATE_RESOLVED,
                message=f"City '{city}' assigned to '{state}' (custom state)",
                details={
                    "city": city,
                    "state": state,
                    "common_states": possible_states,
                },
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.MULTI_STATE_RESOLVED,
                message=f"Multi-state city '{city}' not resolved (no state)",
                details={
                    "city": city,
                    "possible_states": possible_states,
                },
            )

    def _check_component_alignment(
        self,
        validation_result: Any,
        components: dict[str, Any],
        metrics: MatchConfidenceMetrics,
    ) -> GateResult:
        """Check if components align with match."""
        alignment_issues = []
        component_scores: dict[str, float] = {}

        # Get standardized address if available
        standardized = ""
        if validation_result and hasattr(validation_result, "standardized"):
            standardized = (validation_result.standardized or "").upper()

        # Check each component
        city = components.get("city", "").upper()
        state = components.get("state", "").upper()
        zip_code = components.get("zip", "")

        if city:
            if standardized and city in standardized:
                component_scores["city"] = 1.0
            else:
                component_scores["city"] = 0.8  # May be normalized differently
                if standardized and city not in standardized:
                    alignment_issues.append(f"City '{city}' not in standardized address")

        if state:
            if standardized and state in standardized:
                component_scores["state"] = 1.0
            else:
                component_scores["state"] = 0.8

        if zip_code:
            if standardized and zip_code in standardized:
                component_scores["zip"] = 1.0
            else:
                component_scores["zip"] = 0.7
                if standardized and zip_code not in standardized:
                    alignment_issues.append(f"ZIP '{zip_code}' not in standardized address")

        metrics.component_scores = component_scores

        if not alignment_issues:
            return self._gate_pass(
                gate=AddressEnforcerGate.COMPONENT_ALIGNMENT,
                message="Components align with match",
                details={"component_scores": component_scores},
            )
        else:
            # Minor misalignments are okay
            if len(alignment_issues) <= 1:
                return self._gate_pass(
                    gate=AddressEnforcerGate.COMPONENT_ALIGNMENT,
                    message=f"Minor component alignment issues ({len(alignment_issues)})",
                    details={
                        "component_scores": component_scores,
                        "issues": alignment_issues,
                    },
                )
            else:
                return self._gate_fail(
                    gate=AddressEnforcerGate.COMPONENT_ALIGNMENT,
                    message=f"Component alignment issues ({len(alignment_issues)})",
                    details={
                        "component_scores": component_scores,
                        "issues": alignment_issues,
                    },
                )

    def _gate_pass(
        self,
        gate: AddressEnforcerGate,
        message: str,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> GateResult:
        """Create a passing gate result with AddressEnforcerGate."""
        return GateResult(
            gate=gate,  # type: ignore[arg-type]
            passed=True,
            message=message,
            threshold=threshold,
            actual_value=actual_value,
            details=details or {},
        )

    def _gate_fail(
        self,
        gate: AddressEnforcerGate,
        message: str,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> GateResult:
        """Create a failing gate result with AddressEnforcerGate."""
        return GateResult(
            gate=gate,  # type: ignore[arg-type]
            passed=False,
            message=message,
            threshold=threshold,
            actual_value=actual_value,
            details=details or {},
        )
