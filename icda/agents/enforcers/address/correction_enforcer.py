"""Correction Enforcer Agent - Address correction quality gates.

Agent 3 of 6 in the address validation enforcer pipeline.
Validates that address corrections were performed correctly.

Quality Gates:
1. TYPO_DETECTION_VALID - Typos correctly identified
2. TRANSPOSITION_DETECTED - Digit transpositions found (12345→12354)
3. MISSPELLING_RECOVERED - Fuzzy match corrections work
4. CORRECTION_CONFIDENCE - Correction confidence >= 0.75
5. NO_OVER_CORRECTION - Didn't change correct values
6. ORIGINAL_INTENT_PRESERVED - Semantic meaning maintained
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from ..base_enforcer import BaseEnforcer, EnforcerResult, GateResult
from .models import (
    AddressEnforcerGate,
    ComponentType,
    CorrectionDetail,
    CorrectionMetrics,
    CorrectionType,
)


# Common city misspellings database
CITY_CORRECTIONS = {
    "CLEVLAND": "CLEVELAND",
    "MOUNT VERNIN": "MOUNT VERNON",
    "SAN DEIGO": "SAN DIEGO",
    "NEW YROK": "NEW YORK",
    "SAN ANTONIA": "SAN ANTONIO",
    "ATLANA": "ATLANTA",
    "ATHANS": "ATHENS",
    "SACREMENTO": "SACRAMENTO",
    "NAPERVILE": "NAPERVILLE",
    "BELEVUE": "BELLEVUE",
    "DALAS": "DALLAS",
    "SYRACEUSE": "SYRACUSE",
    "SPRING VALEY": "SPRING VALLEY",
    "NORTH LAS VAGAS": "NORTH LAS VEGAS",
    "TAKOMA": "TACOMA",
    "SAN FRANCISO": "SAN FRANCISCO",
    "RALIEGH": "RALEIGH",
    "EIRE": "ERIE",
    "DALLASS": "DALLAS",
    "ATHEN": "ATHENS",
    "SPRIGFIELD": "SPRINGFIELD",
}

# Common street misspellings
STREET_CORRECTIONS = {
    "ASPN": "ASPEN",
    "FORREST": "FOREST",
    "JEFFRSON": "JEFFERSON",
    "MAPEL": "MAPLE",
    "MAPL": "MAPLE",
    "VALEY": "VALLEY",
    "BRODWAY": "BROADWAY",
    "CHESNUT": "CHESTNUT",
    "LINCON": "LINCOLN",
    "KENEDY": "KENNEDY",
    "WILOW": "WILLOW",
    "JAKSON": "JACKSON",
    "SPRNG": "SPRING",
    "WASHINGTN": "WASHINGTON",
}


class CorrectionEnforcerAgent(BaseEnforcer):
    """Enforcer agent for address correction quality validation.

    Validates that typos, transpositions, and misspellings were
    correctly identified and fixed while preserving original intent.
    """

    __slots__ = ("_confidence_threshold",)

    def __init__(
        self,
        enabled: bool = True,
        strict_mode: bool = False,
        confidence_threshold: float = 0.75,
    ):
        """Initialize CorrectionEnforcerAgent.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the entire check.
            confidence_threshold: Minimum confidence for corrections.
        """
        super().__init__(
            name="CorrectionEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._confidence_threshold = confidence_threshold

    def get_gates(self) -> list[AddressEnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            AddressEnforcerGate.TYPO_DETECTION_VALID,
            AddressEnforcerGate.TRANSPOSITION_DETECTED,
            AddressEnforcerGate.MISSPELLING_RECOVERED,
            AddressEnforcerGate.CORRECTION_CONFIDENCE,
            AddressEnforcerGate.NO_OVER_CORRECTION,
            AddressEnforcerGate.ORIGINAL_INTENT_PRESERVED,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run correction quality gates.

        Args:
            context: Dictionary containing:
                - original_address: Original input address string
                - corrected_address: Corrected address string
                - original_components: Original parsed components
                - corrected_components: Components after correction
                - corrections_applied: List of corrections made
                - validation_result: Optional validation result object
                - error_type: Expected error type (for validation)
                - expected_correction: Expected correction details

        Returns:
            EnforcerResult with gate results and correction metrics.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
                metrics={"enforcement_skipped": True},
            )

        # Extract data from context
        original = context.get("original_address", "")
        corrected = context.get("corrected_address", "")
        original_components = context.get("original_components", {})
        corrected_components = context.get("corrected_components", {})
        corrections_applied = context.get("corrections_applied", [])
        error_type = context.get("error_type", "")
        expected = context.get("expected_correction", {})
        validation_result = context.get("validation_result")

        # Use validation result data if available
        if validation_result:
            if hasattr(validation_result, "standardized") and validation_result.standardized:
                corrected = validation_result.standardized
            if hasattr(validation_result, "corrections_applied"):
                corrections_applied = validation_result.corrections_applied or []

        metrics = CorrectionMetrics()
        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Analyze corrections
        self._analyze_corrections(
            original,
            corrected,
            original_components,
            corrected_components,
            error_type,
            expected,
            metrics,
        )

        # Gate 1: Typo Detection Valid
        typo_result = self._check_typo_detection(
            original, corrected, error_type, expected, metrics
        )
        (gates_passed if typo_result.passed else gates_failed).append(typo_result)

        # Gate 2: Transposition Detected
        trans_result = self._check_transposition_detection(
            original, corrected, error_type, expected, metrics
        )
        (gates_passed if trans_result.passed else gates_failed).append(trans_result)

        # Gate 3: Misspelling Recovered
        misspell_result = self._check_misspelling_recovery(
            original, corrected, error_type, expected, metrics
        )
        (gates_passed if misspell_result.passed else gates_failed).append(misspell_result)

        # Gate 4: Correction Confidence
        conf_result = self._check_correction_confidence(context, metrics)
        (gates_passed if conf_result.passed else gates_failed).append(conf_result)

        # Gate 5: No Over-Correction
        over_result = self._check_no_over_correction(
            original, corrected, error_type, expected, metrics
        )
        (gates_passed if over_result.passed else gates_failed).append(over_result)

        # Gate 6: Original Intent Preserved
        intent_result = self._check_intent_preserved(
            original, corrected, error_type, metrics
        )
        (gates_passed if intent_result.passed else gates_failed).append(intent_result)

        # Create result
        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "correction": metrics.to_dict(),
            "total_corrections": metrics.total_corrections,
        }

        return result

    def _analyze_corrections(
        self,
        original: str,
        corrected: str,
        original_components: dict[str, Any],
        corrected_components: dict[str, Any],
        error_type: str,
        expected: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> None:
        """Analyze corrections made and populate metrics."""
        orig_upper = original.upper()
        corr_upper = corrected.upper()

        # Detect city corrections
        orig_city = original_components.get("city", "").upper()
        corr_city = corrected_components.get("city", "").upper()

        if orig_city and corr_city and orig_city != corr_city:
            similarity = SequenceMatcher(None, orig_city, corr_city).ratio()
            correction = CorrectionDetail(
                component=ComponentType.CITY,
                original=orig_city,
                corrected=corr_city,
                correction_type=CorrectionType.MISSPELLING,
                confidence=similarity,
                reason=f"City corrected from '{orig_city}' to '{corr_city}'",
            )
            metrics.corrections.append(correction)
            metrics.misspellings_fixed += 1

        # Detect street corrections
        orig_street = original_components.get("street", "").upper()
        corr_street = corrected_components.get("street", "").upper()

        if orig_street and corr_street and orig_street != corr_street:
            similarity = SequenceMatcher(None, orig_street, corr_street).ratio()

            # Determine correction type
            corr_type = CorrectionType.MISSPELLING
            if self._is_typo(orig_street, corr_street):
                corr_type = CorrectionType.TYPO
                metrics.typos_fixed += 1
            else:
                metrics.misspellings_fixed += 1

            correction = CorrectionDetail(
                component=ComponentType.STREET_NAME,
                original=orig_street,
                corrected=corr_street,
                correction_type=corr_type,
                confidence=similarity,
                reason=f"Street corrected from '{orig_street}' to '{corr_street}'",
            )
            metrics.corrections.append(correction)

        # Detect ZIP corrections
        orig_zip = original_components.get("zip", "")
        corr_zip = corrected_components.get("zip", "")

        if orig_zip and corr_zip and orig_zip != corr_zip:
            # Check if it's a transposition
            if self._is_transposition(orig_zip, corr_zip):
                metrics.transpositions_fixed += 1
                corr_type = CorrectionType.TRANSPOSITION
            else:
                corr_type = CorrectionType.TYPO
                metrics.typos_fixed += 1

            correction = CorrectionDetail(
                component=ComponentType.ZIP,
                original=orig_zip,
                corrected=corr_zip,
                correction_type=corr_type,
                confidence=0.9,
                reason=f"ZIP corrected from '{orig_zip}' to '{corr_zip}'",
            )
            metrics.corrections.append(correction)

        # Calculate overall confidence
        if metrics.corrections:
            metrics.overall_confidence = sum(
                c.confidence for c in metrics.corrections
            ) / len(metrics.corrections)
        else:
            metrics.overall_confidence = 1.0

    def _is_typo(self, original: str, corrected: str) -> bool:
        """Check if the difference is likely a typo (1-2 char difference)."""
        if len(original) != len(corrected):
            return abs(len(original) - len(corrected)) <= 1

        diffs = sum(1 for a, b in zip(original, corrected) if a != b)
        return diffs <= 2

    def _is_transposition(self, original: str, corrected: str) -> bool:
        """Check if the difference is a digit transposition."""
        if len(original) != len(corrected):
            return False

        # Count differences
        diffs = [(i, original[i], corrected[i])
                 for i in range(len(original)) if original[i] != corrected[i]]

        if len(diffs) != 2:
            return False

        # Check if swapped positions
        i1, c1_orig, c1_corr = diffs[0]
        i2, c2_orig, c2_corr = diffs[1]

        return c1_orig == c2_corr and c2_orig == c1_corr

    def _check_typo_detection(
        self,
        original: str,
        corrected: str,
        error_type: str,
        expected: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check if typos were correctly detected and fixed."""
        # If error type isn't a typo, pass by default
        if error_type not in ("misspelled_street", "misspelled_city"):
            return self._gate_pass(
                gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
                message="No typo detection needed for this error type",
                details={"error_type": error_type, "typos_fixed": metrics.typos_fixed},
            )

        # Check if expected correction was applied
        corr_upper = corrected.upper()

        if error_type == "misspelled_city":
            expected_city = expected.get("city", "").upper()
            if expected_city and expected_city in corr_upper:
                return self._gate_pass(
                    gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
                    message=f"City typo correctly fixed to '{expected_city}'",
                    details={"expected": expected_city, "found": True},
                )
            else:
                return self._gate_fail(
                    gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
                    message=f"City typo not corrected (expected: '{expected_city}')",
                    details={"expected": expected_city, "corrected": corrected},
                )

        if error_type == "misspelled_street":
            expected_street = expected.get("street", "").upper()
            if expected_street:
                # Extract street name parts for comparison
                street_parts = expected_street.split()
                if len(street_parts) > 1:
                    street_name = street_parts[1]  # Get name part
                    if street_name in corr_upper:
                        return self._gate_pass(
                            gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
                            message=f"Street typo correctly fixed",
                            details={"expected_name": street_name, "found": True},
                        )

            return self._gate_fail(
                gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
                message=f"Street typo not corrected",
                details={"expected": expected_street, "corrected": corrected},
            )

        return self._gate_pass(
            gate=AddressEnforcerGate.TYPO_DETECTION_VALID,
            message="Typo detection check complete",
            details={"typos_fixed": metrics.typos_fixed},
        )

    def _check_transposition_detection(
        self,
        original: str,
        corrected: str,
        error_type: str,
        expected: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check if digit transpositions were detected."""
        # Only check if error type is transposition
        if error_type != "transposed_zip_digits":
            return self._gate_pass(
                gate=AddressEnforcerGate.TRANSPOSITION_DETECTED,
                message="No transposition detection needed",
                details={"error_type": error_type},
            )

        expected_zip = expected.get("zip", "")
        if expected_zip and expected_zip in corrected:
            metrics.transpositions_fixed += 1
            return self._gate_pass(
                gate=AddressEnforcerGate.TRANSPOSITION_DETECTED,
                message=f"ZIP transposition corrected to '{expected_zip}'",
                details={
                    "expected_zip": expected_zip,
                    "transpositions_fixed": metrics.transpositions_fixed,
                },
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.TRANSPOSITION_DETECTED,
                message=f"ZIP transposition not corrected",
                details={"expected_zip": expected_zip, "corrected": corrected},
            )

    def _check_misspelling_recovery(
        self,
        original: str,
        corrected: str,
        error_type: str,
        expected: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check if misspellings were recovered through fuzzy matching."""
        if error_type not in ("misspelled_city", "misspelled_street"):
            return self._gate_pass(
                gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
                message="No misspelling recovery needed",
                details={"error_type": error_type},
            )

        corr_upper = corrected.upper()
        orig_upper = original.upper()

        if error_type == "misspelled_city":
            expected_city = expected.get("city", "").upper()

            # Check known corrections
            for misspelled, correct in CITY_CORRECTIONS.items():
                if misspelled in orig_upper:
                    if correct in corr_upper:
                        return self._gate_pass(
                            gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
                            message=f"City misspelling recovered: {misspelled}→{correct}",
                            details={"original": misspelled, "corrected": correct},
                        )

            # Check if expected correction is present
            if expected_city and expected_city in corr_upper:
                return self._gate_pass(
                    gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
                    message=f"City misspelling recovered to '{expected_city}'",
                    details={"expected": expected_city},
                )

        if error_type == "misspelled_street":
            expected_street = expected.get("street", "").upper()

            # Check known corrections
            for misspelled, correct in STREET_CORRECTIONS.items():
                if misspelled in orig_upper:
                    if correct in corr_upper:
                        return self._gate_pass(
                            gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
                            message=f"Street misspelling recovered: {misspelled}→{correct}",
                            details={"original": misspelled, "corrected": correct},
                        )

            # Check if expected street is present
            if expected_street:
                parts = expected_street.split()
                if len(parts) > 1 and parts[1] in corr_upper:
                    return self._gate_pass(
                        gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
                        message="Street misspelling recovered",
                        details={"expected": expected_street},
                    )

        return self._gate_fail(
            gate=AddressEnforcerGate.MISSPELLING_RECOVERED,
            message="Misspelling not recovered",
            details={
                "error_type": error_type,
                "expected": expected,
                "corrected": corrected,
            },
        )

    def _check_correction_confidence(
        self,
        context: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check if correction confidence meets threshold."""
        # Get confidence from validation result or use calculated
        validation_result = context.get("validation_result")
        if validation_result and hasattr(validation_result, "overall_confidence"):
            confidence = validation_result.overall_confidence
        else:
            confidence = metrics.overall_confidence

        metrics.overall_confidence = confidence

        if confidence >= self._confidence_threshold:
            return self._gate_pass(
                gate=AddressEnforcerGate.CORRECTION_CONFIDENCE,
                message=f"Correction confidence {confidence:.1%} meets threshold",
                threshold=self._confidence_threshold,
                actual_value=confidence,
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.CORRECTION_CONFIDENCE,
                message=f"Correction confidence {confidence:.1%} below threshold",
                threshold=self._confidence_threshold,
                actual_value=confidence,
            )

    def _check_no_over_correction(
        self,
        original: str,
        corrected: str,
        error_type: str,
        expected: dict[str, Any],
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check that we didn't over-correct (change things that were correct)."""
        over_corrections = []

        orig_upper = original.upper()
        corr_upper = corrected.upper()

        # For certain error types, check that unrelated parts weren't changed
        if error_type == "misspelled_city":
            # Street should not have changed significantly
            orig_words = orig_upper.split(",")[0].split() if "," in orig_upper else orig_upper.split()[:4]
            corr_words = corr_upper.split(",")[0].split() if "," in corr_upper else corr_upper.split()[:4]

            # Check street number preservation
            if orig_words and corr_words:
                if orig_words[0].isdigit() and corr_words[0].isdigit():
                    if orig_words[0] != corr_words[0]:
                        over_corrections.append(
                            f"Street number changed: {orig_words[0]}→{corr_words[0]}"
                        )

        if error_type == "transposed_zip_digits":
            # City and state should not have changed
            # This is harder to check without proper parsing
            pass

        metrics.over_corrections = len(over_corrections)

        if not over_corrections:
            return self._gate_pass(
                gate=AddressEnforcerGate.NO_OVER_CORRECTION,
                message="No over-corrections detected",
                details={"over_corrections": 0},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.NO_OVER_CORRECTION,
                message=f"Found {len(over_corrections)} over-correction(s)",
                details={"over_corrections": over_corrections},
            )

    def _check_intent_preserved(
        self,
        original: str,
        corrected: str,
        error_type: str,
        metrics: CorrectionMetrics,
    ) -> GateResult:
        """Check that original intent/meaning was preserved."""
        orig_upper = original.upper()
        corr_upper = corrected.upper()

        # Calculate overall similarity
        similarity = SequenceMatcher(None, orig_upper, corr_upper).ratio()

        # Intent is considered preserved if similarity is high enough
        # (corrections should be minor changes, not rewrites)
        min_similarity = 0.6  # Allow for significant corrections

        # Extract core components for comparison
        orig_numbers = re.findall(r'\d+', orig_upper)
        corr_numbers = re.findall(r'\d+', corr_upper)

        # Street number should be preserved (first number)
        number_preserved = True
        if orig_numbers and corr_numbers:
            if orig_numbers[0] != corr_numbers[0]:
                number_preserved = False

        metrics.intent_preserved = similarity >= min_similarity and number_preserved

        if metrics.intent_preserved:
            return self._gate_pass(
                gate=AddressEnforcerGate.ORIGINAL_INTENT_PRESERVED,
                message=f"Original intent preserved (similarity: {similarity:.1%})",
                details={
                    "similarity": similarity,
                    "number_preserved": number_preserved,
                },
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.ORIGINAL_INTENT_PRESERVED,
                message=f"Original intent may not be preserved (similarity: {similarity:.1%})",
                details={
                    "similarity": similarity,
                    "number_preserved": number_preserved,
                    "original": original[:50],
                    "corrected": corrected[:50],
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
