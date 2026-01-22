"""Completion Enforcer Agent - Address completion quality gates.

Agent 2 of 6 in the address validation enforcer pipeline.
Validates that address completion/inference was performed correctly.

Quality Gates:
1. ZIP_INFERENCE_VALID - City/state from ZIP correct
2. CITY_INFERENCE_VALID - City matches ZIP database
3. STATE_INFERENCE_VALID - State matches ZIP/city
4. COMPLETION_CONFIDENCE - Confidence >= 0.70
5. NO_CONFLICTING_INFERENCES - No contradictory completions
6. COMPONENT_CHAIN_VALID - Completion chain consistent
"""

from __future__ import annotations

from typing import Any

from ..base_enforcer import BaseEnforcer, EnforcerResult, GateResult
from .models import AddressEnforcerGate, CompletionMetrics


# ZIP code to city/state mapping (sample data)
# In production, this would come from a database
ZIP_DATABASE = {
    "60535": {"city": "AURORA", "state": "IL"},
    "44154": {"city": "CLEVELAND", "state": "OH"},
    "98065": {"city": "RENTON", "state": "WA"},
    "10551": {"city": "MOUNT VERNON", "state": "NY"},
    "27607": {"city": "RALEIGH", "state": "NC"},
    "94162": {"city": "SAN FRANCISCO", "state": "CA"},
    "92168": {"city": "SAN DIEGO", "state": "CA"},
    "89158": {"city": "PARADISE", "state": "NV"},
    "16561": {"city": "ERIE", "state": "PA"},
    "98089": {"city": "KENT", "state": "WA"},
    "85366": {"city": "PEORIA", "state": "AZ"},
    "92574": {"city": "RIVERSIDE", "state": "CA"},
    "30618": {"city": "ATHENS", "state": "GA"},
    "13279": {"city": "SYRACUSE", "state": "NY"},
    "30370": {"city": "ATLANTA", "state": "GA"},
    "89141": {"city": "SPRING VALLEY", "state": "NV"},
    "98003": {"city": "FEDERAL WAY", "state": "WA"},
    "10008": {"city": "NEW YORK", "state": "NY"},
    "23573": {"city": "NORFOLK", "state": "VA"},
    "30906": {"city": "AUGUSTA", "state": "GA"},
    "78261": {"city": "SAN ANTONIO", "state": "TX"},
    "75230": {"city": "DALLAS", "state": "TX"},
    "30386": {"city": "ATLANTA", "state": "GA"},
    "90809": {"city": "LONG BEACH", "state": "CA"},
    "89430": {"city": "SPARKS", "state": "NV"},
    "95835": {"city": "SACRAMENTO", "state": "CA"},
    "89086": {"city": "NORTH LAS VEGAS", "state": "NV"},
    "16538": {"city": "ERIE", "state": "PA"},
    "18545": {"city": "SCRANTON", "state": "PA"},
    "89184": {"city": "PARADISE", "state": "NV"},
    "33027": {"city": "HIALEAH", "state": "FL"},
    "60552": {"city": "NAPERVILLE", "state": "IL"},
    "30603": {"city": "ATHENS", "state": "GA"},
    "30627": {"city": "ATHENS", "state": "GA"},
    "30957": {"city": "AUGUSTA", "state": "GA"},
    "89577": {"city": "RENO", "state": "NV"},
    "75286": {"city": "DALLAS", "state": "TX"},
    "92118": {"city": "SAN DIEGO", "state": "CA"},
    "18572": {"city": "SCRANTON", "state": "PA"},
    "98015": {"city": "BELLEVUE", "state": "WA"},
    "98436": {"city": "TACOMA", "state": "WA"},
    "99209": {"city": "SPOKANE", "state": "WA"},
    "98216": {"city": "EVERETT", "state": "WA"},
}


class CompletionEnforcerAgent(BaseEnforcer):
    """Enforcer agent for address completion quality validation.

    Validates that address components were correctly inferred from
    partial information (e.g., ZIP→city, city→state).
    """

    __slots__ = ("_confidence_threshold",)

    def __init__(
        self,
        enabled: bool = True,
        strict_mode: bool = False,
        confidence_threshold: float = 0.70,
    ):
        """Initialize CompletionEnforcerAgent.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the entire check.
            confidence_threshold: Minimum confidence for completions.
        """
        super().__init__(
            name="CompletionEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._confidence_threshold = confidence_threshold

    def get_gates(self) -> list[AddressEnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            AddressEnforcerGate.ZIP_INFERENCE_VALID,
            AddressEnforcerGate.CITY_INFERENCE_VALID,
            AddressEnforcerGate.STATE_INFERENCE_VALID,
            AddressEnforcerGate.COMPLETION_CONFIDENCE,
            AddressEnforcerGate.NO_CONFLICTING_INFERENCES,
            AddressEnforcerGate.COMPONENT_CHAIN_VALID,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run completion quality gates.

        Args:
            context: Dictionary containing:
                - original_address: Original input address string
                - original_components: Original parsed components
                - completed_components: Components after completion
                - completions_applied: List of completions made
                - validation_result: Optional validation result object

        Returns:
            EnforcerResult with gate results and completion metrics.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
                metrics={"enforcement_skipped": True},
            )

        # Extract data from context
        original_components = context.get("original_components", {})
        completed_components = context.get("completed_components", {})
        completions_applied = context.get("completions_applied", [])
        validation_result = context.get("validation_result")

        # If validation result has completions, use those
        if validation_result and hasattr(validation_result, "completions_applied"):
            completions_applied = validation_result.completions_applied or []

        metrics = CompletionMetrics()
        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Determine what was inferred
        self._analyze_inferences(
            original_components,
            completed_components,
            completions_applied,
            metrics,
        )

        # Gate 1: ZIP Inference Valid
        zip_result = self._check_zip_inference(completed_components, metrics)
        (gates_passed if zip_result.passed else gates_failed).append(zip_result)

        # Gate 2: City Inference Valid
        city_result = self._check_city_inference(completed_components, metrics)
        (gates_passed if city_result.passed else gates_failed).append(city_result)

        # Gate 3: State Inference Valid
        state_result = self._check_state_inference(completed_components, metrics)
        (gates_passed if state_result.passed else gates_failed).append(state_result)

        # Gate 4: Completion Confidence
        conf_result = self._check_completion_confidence(context, metrics)
        (gates_passed if conf_result.passed else gates_failed).append(conf_result)

        # Gate 5: No Conflicting Inferences
        conflict_result = self._check_no_conflicts(completed_components, metrics)
        (gates_passed if conflict_result.passed else gates_failed).append(conflict_result)

        # Gate 6: Component Chain Valid
        chain_result = self._check_component_chain(completed_components, metrics)
        (gates_passed if chain_result.passed else gates_failed).append(chain_result)

        # Create result
        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "completion": metrics.to_dict(),
            "has_inferences": metrics.has_inferences,
        }

        return result

    def _analyze_inferences(
        self,
        original: dict[str, Any],
        completed: dict[str, Any],
        completions: list[str],
        metrics: CompletionMetrics,
    ) -> None:
        """Analyze what inferences were made."""
        orig_zip = original.get("zip", "")
        orig_city = original.get("city", "")
        orig_state = original.get("state", "")

        comp_zip = completed.get("zip", "")
        comp_city = completed.get("city", "")
        comp_state = completed.get("state", "")

        # Check for ZIP inference
        if not orig_zip and comp_zip:
            metrics.zip_inferred = True
            metrics.inferred_zip = comp_zip
            metrics.inference_chain.append(f"ZIP inferred: {comp_zip}")

        # Check for city inference
        if not orig_city and comp_city:
            metrics.city_inferred = True
            metrics.inferred_city = comp_city
            metrics.inference_chain.append(f"City inferred: {comp_city}")

        # Check for state inference
        if not orig_state and comp_state:
            metrics.state_inferred = True
            metrics.inferred_state = comp_state
            metrics.inference_chain.append(f"State inferred: {comp_state}")

        # Parse completions list for additional details
        for completion in completions:
            comp_lower = completion.lower()
            if "zip" in comp_lower and not metrics.zip_inferred:
                metrics.zip_inferred = True
            if "city" in comp_lower and not metrics.city_inferred:
                metrics.city_inferred = True
            if "state" in comp_lower and not metrics.state_inferred:
                metrics.state_inferred = True

    def _check_zip_inference(
        self,
        components: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check if ZIP was correctly inferred from city/state."""
        if not metrics.zip_inferred:
            return self._gate_pass(
                gate=AddressEnforcerGate.ZIP_INFERENCE_VALID,
                message="No ZIP inference needed",
                details={"inferred": False},
            )

        inferred_zip = metrics.inferred_zip or components.get("zip", "")
        city = components.get("city", "").upper()
        state = components.get("state", "").upper()

        # Validate against database
        if inferred_zip in ZIP_DATABASE:
            db_entry = ZIP_DATABASE[inferred_zip]
            city_match = city in db_entry["city"] or db_entry["city"] in city
            state_match = state == db_entry["state"]

            if city_match and state_match:
                return self._gate_pass(
                    gate=AddressEnforcerGate.ZIP_INFERENCE_VALID,
                    message=f"ZIP {inferred_zip} correctly matches city/state",
                    details={
                        "inferred_zip": inferred_zip,
                        "expected_city": db_entry["city"],
                        "expected_state": db_entry["state"],
                    },
                )
            else:
                return self._gate_fail(
                    gate=AddressEnforcerGate.ZIP_INFERENCE_VALID,
                    message=f"ZIP {inferred_zip} does not match city/state",
                    details={
                        "inferred_zip": inferred_zip,
                        "provided_city": city,
                        "provided_state": state,
                        "expected_city": db_entry["city"],
                        "expected_state": db_entry["state"],
                    },
                )

        # ZIP not in database - pass with warning
        return self._gate_pass(
            gate=AddressEnforcerGate.ZIP_INFERENCE_VALID,
            message=f"ZIP {inferred_zip} inferred (not verified against database)",
            details={"inferred_zip": inferred_zip, "verified": False},
        )

    def _check_city_inference(
        self,
        components: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check if city was correctly inferred from ZIP."""
        if not metrics.city_inferred:
            return self._gate_pass(
                gate=AddressEnforcerGate.CITY_INFERENCE_VALID,
                message="No city inference needed",
                details={"inferred": False},
            )

        inferred_city = metrics.inferred_city or components.get("city", "")
        zip_code = components.get("zip", "")

        if zip_code and zip_code in ZIP_DATABASE:
            expected_city = ZIP_DATABASE[zip_code]["city"]
            inferred_upper = inferred_city.upper()

            if inferred_upper == expected_city or expected_city in inferred_upper:
                return self._gate_pass(
                    gate=AddressEnforcerGate.CITY_INFERENCE_VALID,
                    message=f"City '{inferred_city}' correctly inferred from ZIP {zip_code}",
                    details={
                        "inferred_city": inferred_city,
                        "zip_code": zip_code,
                        "expected": expected_city,
                    },
                )
            else:
                return self._gate_fail(
                    gate=AddressEnforcerGate.CITY_INFERENCE_VALID,
                    message=f"City '{inferred_city}' incorrect for ZIP {zip_code}",
                    details={
                        "inferred_city": inferred_city,
                        "zip_code": zip_code,
                        "expected": expected_city,
                    },
                )

        # No ZIP to validate against
        return self._gate_pass(
            gate=AddressEnforcerGate.CITY_INFERENCE_VALID,
            message=f"City '{inferred_city}' inferred (no ZIP for validation)",
            details={"inferred_city": inferred_city, "verified": False},
        )

    def _check_state_inference(
        self,
        components: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check if state was correctly inferred from ZIP/city."""
        if not metrics.state_inferred:
            return self._gate_pass(
                gate=AddressEnforcerGate.STATE_INFERENCE_VALID,
                message="No state inference needed",
                details={"inferred": False},
            )

        inferred_state = metrics.inferred_state or components.get("state", "")
        zip_code = components.get("zip", "")

        if zip_code and zip_code in ZIP_DATABASE:
            expected_state = ZIP_DATABASE[zip_code]["state"]
            inferred_upper = inferred_state.upper()

            if inferred_upper == expected_state:
                return self._gate_pass(
                    gate=AddressEnforcerGate.STATE_INFERENCE_VALID,
                    message=f"State '{inferred_state}' correctly inferred from ZIP {zip_code}",
                    details={
                        "inferred_state": inferred_state,
                        "zip_code": zip_code,
                        "expected": expected_state,
                    },
                )
            else:
                return self._gate_fail(
                    gate=AddressEnforcerGate.STATE_INFERENCE_VALID,
                    message=f"State '{inferred_state}' incorrect for ZIP {zip_code}",
                    details={
                        "inferred_state": inferred_state,
                        "zip_code": zip_code,
                        "expected": expected_state,
                    },
                )

        # No ZIP to validate against
        return self._gate_pass(
            gate=AddressEnforcerGate.STATE_INFERENCE_VALID,
            message=f"State '{inferred_state}' inferred (no ZIP for validation)",
            details={"inferred_state": inferred_state, "verified": False},
        )

    def _check_completion_confidence(
        self,
        context: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check if completion confidence meets threshold."""
        # Get confidence from validation result or context
        validation_result = context.get("validation_result")
        if validation_result and hasattr(validation_result, "overall_confidence"):
            metrics.confidence = validation_result.overall_confidence
        else:
            metrics.confidence = context.get("confidence", 0.85)

        if metrics.confidence >= self._confidence_threshold:
            return self._gate_pass(
                gate=AddressEnforcerGate.COMPLETION_CONFIDENCE,
                message=f"Completion confidence {metrics.confidence:.1%} meets threshold",
                threshold=self._confidence_threshold,
                actual_value=metrics.confidence,
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.COMPLETION_CONFIDENCE,
                message=f"Completion confidence {metrics.confidence:.1%} below threshold",
                threshold=self._confidence_threshold,
                actual_value=metrics.confidence,
            )

    def _check_no_conflicts(
        self,
        components: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check for conflicting inferences."""
        conflicts = []

        zip_code = components.get("zip", "")
        city = components.get("city", "").upper()
        state = components.get("state", "").upper()

        if zip_code and zip_code in ZIP_DATABASE:
            db_entry = ZIP_DATABASE[zip_code]

            # Check city conflict
            if city and city not in db_entry["city"] and db_entry["city"] not in city:
                conflicts.append(
                    f"City '{city}' conflicts with ZIP {zip_code} "
                    f"(expected: {db_entry['city']})"
                )

            # Check state conflict
            if state and state != db_entry["state"]:
                conflicts.append(
                    f"State '{state}' conflicts with ZIP {zip_code} "
                    f"(expected: {db_entry['state']})"
                )

        metrics.conflicts_found = len(conflicts)
        metrics.conflict_details = conflicts

        if not conflicts:
            return self._gate_pass(
                gate=AddressEnforcerGate.NO_CONFLICTING_INFERENCES,
                message="No conflicting inferences found",
                details={"conflicts": 0},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.NO_CONFLICTING_INFERENCES,
                message=f"Found {len(conflicts)} conflicting inference(s)",
                details={"conflicts": conflicts},
            )

    def _check_component_chain(
        self,
        components: dict[str, Any],
        metrics: CompletionMetrics,
    ) -> GateResult:
        """Check that completion chain is consistent."""
        # Verify that inferred components form a valid chain
        zip_code = components.get("zip", "")
        city = components.get("city", "")
        state = components.get("state", "")

        chain_valid = True
        chain_details = []

        # If we have all three, verify they're consistent
        if zip_code and city and state:
            if zip_code in ZIP_DATABASE:
                db_entry = ZIP_DATABASE[zip_code]
                city_upper = city.upper()
                state_upper = state.upper()

                if city_upper not in db_entry["city"] and db_entry["city"] not in city_upper:
                    chain_valid = False
                    chain_details.append(f"City '{city}' not in chain for ZIP {zip_code}")

                if state_upper != db_entry["state"]:
                    chain_valid = False
                    chain_details.append(f"State '{state}' not in chain for ZIP {zip_code}")
            else:
                # ZIP not in database, can't validate chain
                chain_details.append(f"ZIP {zip_code} not in database for chain validation")

        metrics.inference_chain.extend(chain_details)

        if chain_valid:
            return self._gate_pass(
                gate=AddressEnforcerGate.COMPONENT_CHAIN_VALID,
                message="Component chain is consistent",
                details={"chain_details": chain_details},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.COMPONENT_CHAIN_VALID,
                message="Component chain has inconsistencies",
                details={"chain_details": chain_details},
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
