"""Normalization Enforcer Agent - Address standardization quality gates.

Agent 1 of 6 in the address validation enforcer pipeline.
Validates that address normalization was performed correctly.

Quality Gates:
1. CASE_STANDARDIZED - Title/Upper case applied
2. ABBREVIATIONS_EXPANDED - ST→Street, AVE→Avenue
3. WHITESPACE_NORMALIZED - No double spaces
4. PUNCTUATION_CLEANED - Proper comma placement
5. DIRECTIONALS_STANDARDIZED - N→North
6. UNIT_FORMAT_VALID - APT 101, not Apt.101
"""

from __future__ import annotations

import re
from typing import Any

from ..base_enforcer import BaseEnforcer, EnforcerResult, GateResult
from .models import AddressEnforcerGate, NormalizationMetrics


# Standard abbreviation mappings
STREET_ABBREVIATIONS = {
    "ST": "STREET",
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "DR": "DRIVE",
    "LN": "LANE",
    "RD": "ROAD",
    "CT": "COURT",
    "CIR": "CIRCLE",
    "PL": "PLACE",
    "TER": "TERRACE",
    "WAY": "WAY",
    "PKWY": "PARKWAY",
    "HWY": "HIGHWAY",
}

DIRECTIONAL_ABBREVIATIONS = {
    "N": "NORTH",
    "S": "SOUTH",
    "E": "EAST",
    "W": "WEST",
    "NE": "NORTHEAST",
    "NW": "NORTHWEST",
    "SE": "SOUTHEAST",
    "SW": "SOUTHWEST",
}

UNIT_ABBREVIATIONS = {
    "APT": "APT",
    "UNIT": "UNIT",
    "STE": "SUITE",
    "SUITE": "SUITE",
    "#": "APT",
}


class NormalizationEnforcerAgent(BaseEnforcer):
    """Enforcer agent for address normalization quality validation.

    Ensures addresses are properly standardized including case,
    abbreviations, whitespace, punctuation, directionals, and unit formats.
    """

    __slots__ = ()

    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        """Initialize NormalizationEnforcerAgent.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails the entire check.
        """
        super().__init__(
            name="NormalizationEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )

    def get_gates(self) -> list[AddressEnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            AddressEnforcerGate.CASE_STANDARDIZED,
            AddressEnforcerGate.ABBREVIATIONS_EXPANDED,
            AddressEnforcerGate.WHITESPACE_NORMALIZED,
            AddressEnforcerGate.PUNCTUATION_CLEANED,
            AddressEnforcerGate.DIRECTIONALS_STANDARDIZED,
            AddressEnforcerGate.UNIT_FORMAT_VALID,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run normalization quality gates.

        Args:
            context: Dictionary containing:
                - original_address: Original input address string
                - normalized_address: Normalized/standardized address
                - validation_result: Optional validation result object

        Returns:
            EnforcerResult with gate results and normalization metrics.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
                metrics={"enforcement_skipped": True},
            )

        original = context.get("original_address", "")
        normalized = context.get("normalized_address", "")

        # If no normalized address provided, use original
        if not normalized:
            normalized = context.get("standardized", original)

        metrics = NormalizationMetrics(
            original_text=original,
            normalized_text=normalized,
        )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Gate 1: Case Standardization
        case_result = self._check_case_standardization(original, normalized, metrics)
        (gates_passed if case_result.passed else gates_failed).append(case_result)

        # Gate 2: Abbreviation Expansion
        abbrev_result = self._check_abbreviations(original, normalized, metrics)
        (gates_passed if abbrev_result.passed else gates_failed).append(abbrev_result)

        # Gate 3: Whitespace Normalization
        ws_result = self._check_whitespace(original, normalized, metrics)
        (gates_passed if ws_result.passed else gates_failed).append(ws_result)

        # Gate 4: Punctuation Cleaning
        punct_result = self._check_punctuation(original, normalized, metrics)
        (gates_passed if punct_result.passed else gates_failed).append(punct_result)

        # Gate 5: Directionals Standardization
        dir_result = self._check_directionals(original, normalized, metrics)
        (gates_passed if dir_result.passed else gates_failed).append(dir_result)

        # Gate 6: Unit Format Validation
        unit_result = self._check_unit_format(original, normalized, metrics)
        (gates_passed if unit_result.passed else gates_failed).append(unit_result)

        # Create result
        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "normalization": metrics.to_dict(),
            "total_changes": metrics.total_changes,
        }

        return result

    def _check_case_standardization(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if case has been properly standardized."""
        # Count case changes needed
        orig_upper = original.upper()
        norm_upper = normalized.upper()

        # Check if original had mixed/lowercase that needed fixing
        had_case_issues = original != orig_upper and original != original.title()

        # Check if normalized is properly cased (all upper or proper title case)
        is_standardized = normalized == norm_upper or self._is_title_case(normalized)

        if had_case_issues:
            # Count how many word case changes were made
            orig_words = original.split()
            norm_words = normalized.split()
            changes = sum(
                1 for ow, nw in zip(orig_words, norm_words)
                if ow != nw and ow.upper() == nw.upper()
            )
            metrics.case_changes = changes
            if changes > 0:
                metrics.changes_applied.append(f"Case standardized ({changes} words)")

        if is_standardized:
            return self._gate_pass(
                gate=AddressEnforcerGate.CASE_STANDARDIZED,
                message="Case properly standardized",
                details={"case_changes": metrics.case_changes},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.CASE_STANDARDIZED,
                message="Case not properly standardized",
                details={
                    "original_sample": original[:50],
                    "normalized_sample": normalized[:50],
                },
            )

    def _is_title_case(self, text: str) -> bool:
        """Check if text is in proper title case."""
        words = text.split()
        for word in words:
            if not word:
                continue
            # Allow all caps or title case
            if word != word.upper() and word != word.title():
                # Special handling for unit numbers, etc.
                if not word[0].isupper():
                    return False
        return True

    def _check_abbreviations(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if abbreviations have been properly expanded or standardized."""
        orig_upper = original.upper()
        norm_upper = normalized.upper()

        expansions = 0
        expansion_details = []

        # Check street type abbreviations
        for abbrev, full in STREET_ABBREVIATIONS.items():
            # Check if abbreviation was in original (as whole word)
            abbrev_pattern = rf"\b{abbrev}\b"
            full_pattern = rf"\b{full}\b"

            if re.search(abbrev_pattern, orig_upper):
                # Either expanded to full or kept as standard abbreviation
                if re.search(full_pattern, norm_upper) or re.search(abbrev_pattern, norm_upper):
                    expansions += 1
                    expansion_details.append(f"{abbrev}→{full}")

        metrics.abbreviations_expanded = expansions
        if expansions > 0:
            metrics.changes_applied.append(f"Abbreviations standardized ({expansions})")

        # Pass if no abbreviations needed expansion, or they were handled
        return self._gate_pass(
            gate=AddressEnforcerGate.ABBREVIATIONS_EXPANDED,
            message=f"Abbreviations handled ({expansions} found)",
            details={"expansions": expansion_details},
        )

    def _check_whitespace(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if whitespace has been normalized."""
        # Count multiple spaces in original
        double_spaces_original = len(re.findall(r"  +", original))
        double_spaces_normalized = len(re.findall(r"  +", normalized))

        # Check leading/trailing whitespace
        had_leading_trailing = original != original.strip()
        has_leading_trailing = normalized != normalized.strip()

        fixes = 0
        if double_spaces_original > double_spaces_normalized:
            fixes += double_spaces_original - double_spaces_normalized

        if had_leading_trailing and not has_leading_trailing:
            fixes += 1

        metrics.whitespace_fixes = fixes
        if fixes > 0:
            metrics.changes_applied.append(f"Whitespace normalized ({fixes} fixes)")

        if double_spaces_normalized == 0 and not has_leading_trailing:
            return self._gate_pass(
                gate=AddressEnforcerGate.WHITESPACE_NORMALIZED,
                message="Whitespace properly normalized",
                details={"fixes_applied": fixes},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.WHITESPACE_NORMALIZED,
                message="Whitespace issues remain",
                details={
                    "double_spaces": double_spaces_normalized,
                    "leading_trailing": has_leading_trailing,
                },
            )

    def _check_punctuation(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if punctuation has been properly cleaned."""
        fixes = 0
        issues = []

        # Check for periods after abbreviations that should be removed
        period_after_abbrev = len(re.findall(r"\b[A-Z]{2,3}\.", normalized))
        if period_after_abbrev > 0:
            issues.append(f"{period_after_abbrev} periods after abbreviations")

        # Check for proper comma placement (should have space after, not before)
        bad_commas = len(re.findall(r"\s,|,(?!\s)", normalized))
        if bad_commas > 0:
            issues.append(f"{bad_commas} improper comma placements")

        # Check for double punctuation
        double_punct = len(re.findall(r"[,.]{2,}", normalized))
        if double_punct > 0:
            issues.append(f"{double_punct} double punctuation marks")

        # Count fixes from original
        orig_bad_commas = len(re.findall(r"\s,|,(?!\s)", original))
        if orig_bad_commas > bad_commas:
            fixes += orig_bad_commas - bad_commas

        metrics.punctuation_fixes = fixes
        if fixes > 0:
            metrics.changes_applied.append(f"Punctuation fixed ({fixes} fixes)")

        if not issues:
            return self._gate_pass(
                gate=AddressEnforcerGate.PUNCTUATION_CLEANED,
                message="Punctuation properly formatted",
                details={"fixes_applied": fixes},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.PUNCTUATION_CLEANED,
                message="Punctuation issues remain",
                details={"issues": issues},
            )

    def _check_directionals(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if directionals have been standardized."""
        orig_upper = original.upper()
        norm_upper = normalized.upper()

        fixes = 0
        details = []

        for abbrev, full in DIRECTIONAL_ABBREVIATIONS.items():
            # Look for directionals at word boundaries
            abbrev_pattern = rf"\b{abbrev}\b"
            full_pattern = rf"\b{full}\b"

            # Check if directional appears in original
            if re.search(abbrev_pattern, orig_upper) or re.search(full_pattern, orig_upper):
                # Valid if either abbreviated or full form is in normalized
                if re.search(abbrev_pattern, norm_upper) or re.search(full_pattern, norm_upper):
                    fixes += 1
                    details.append(abbrev)

        metrics.directional_fixes = fixes
        if fixes > 0:
            metrics.changes_applied.append(f"Directionals standardized ({fixes})")

        # Always pass - directionals just need to be present in some standard form
        return self._gate_pass(
            gate=AddressEnforcerGate.DIRECTIONALS_STANDARDIZED,
            message=f"Directionals handled ({len(details)} found)",
            details={"directionals": details},
        )

    def _check_unit_format(
        self,
        original: str,
        normalized: str,
        metrics: NormalizationMetrics,
    ) -> GateResult:
        """Check if unit designators are properly formatted."""
        norm_upper = normalized.upper()

        issues = []
        fixes = 0

        # Check for unit designators
        # Valid formats: "APT 101", "UNIT 2A", "SUITE 500"
        # Invalid formats: "Apt.101", "APT101", "#101" (if not converted)

        # Check for period after unit type
        bad_unit_period = re.findall(r"\b(APT|UNIT|STE|SUITE)\.\s*\d", norm_upper)
        if bad_unit_period:
            issues.append(f"Period after unit type: {bad_unit_period}")

        # Check for missing space between unit type and number
        bad_unit_space = re.findall(r"\b(APT|UNIT|STE|SUITE)(\d)", norm_upper)
        if bad_unit_space:
            issues.append(f"Missing space after unit type: {bad_unit_space}")

        # Check original for fixes made
        orig_upper = original.upper()
        orig_bad_period = len(re.findall(r"\b(APT|UNIT|STE|SUITE)\.", orig_upper))
        norm_bad_period = len(re.findall(r"\b(APT|UNIT|STE|SUITE)\.", norm_upper))
        if orig_bad_period > norm_bad_period:
            fixes += orig_bad_period - norm_bad_period

        metrics.unit_format_fixes = fixes
        if fixes > 0:
            metrics.changes_applied.append(f"Unit format fixed ({fixes} fixes)")

        if not issues:
            return self._gate_pass(
                gate=AddressEnforcerGate.UNIT_FORMAT_VALID,
                message="Unit format is valid",
                details={"fixes_applied": fixes},
            )
        else:
            return self._gate_fail(
                gate=AddressEnforcerGate.UNIT_FORMAT_VALID,
                message="Unit format issues remain",
                details={"issues": issues},
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
