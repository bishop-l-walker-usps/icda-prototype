"""Quality enforcer - Stage 4.

Scores address quality and flags low-confidence records.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    BaseIngestionEnforcer,
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)

if TYPE_CHECKING:
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class QualityEnforcer(BaseIngestionEnforcer):
    """Stage 4: Scores address quality, flags low-confidence.

    Gates:
    - COMPLETENESS_SCORE: Address has sufficient components
    - CONFIDENCE_THRESHOLD: Parsing confidence is adequate
    - PR_URBANIZATION: PR addresses have urbanization
    - NO_INVALID_CHARS: No invalid characters in address
    """

    __slots__ = (
        "_min_completeness",
        "_min_confidence",
        "_require_pr_urbanization",
    )

    # Component weights for completeness scoring
    COMPONENT_WEIGHTS = {
        "street_number": 0.15,
        "street_name": 0.25,
        "city": 0.15,
        "state": 0.15,
        "zip_code": 0.20,
        "unit": 0.05,
        "urbanization": 0.05,
    }

    def __init__(
        self,
        min_completeness: float = 0.6,
        min_confidence: float = 0.7,
        require_pr_urbanization: bool = True,
        enabled: bool = True,
    ):
        """Initialize quality enforcer.

        Args:
            min_completeness: Minimum completeness score (0.0-1.0).
            min_confidence: Minimum confidence threshold.
            require_pr_urbanization: Whether PR addresses need urbanization.
            enabled: Whether enforcer is active.
        """
        super().__init__("quality_enforcer", enabled)
        self._min_completeness = min_completeness
        self._min_confidence = min_confidence
        self._require_pr_urbanization = require_pr_urbanization

    def get_gates(self) -> list[IngestionGate]:
        """Get gates this enforcer checks."""
        return [
            IngestionGate.COMPLETENESS_SCORE,
            IngestionGate.CONFIDENCE_THRESHOLD,
            IngestionGate.PR_URBANIZATION,
            IngestionGate.NO_INVALID_CHARS,
        ]

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Score address quality.

        Args:
            record: IngestionRecord to validate.
            context: Additional context.

        Returns:
            IngestionEnforcerResult with quality score.
        """
        gates_passed: list[IngestionGateResult] = []
        gates_failed: list[IngestionGateResult] = []

        addr = record.parsed_address

        if not addr:
            # No address - fail all quality gates
            for gate in self.get_gates():
                gates_failed.append(
                    self._gate_fail(gate, "No parsed address to evaluate")
                )
            return self._create_result(gates_passed, gates_failed)

        # Gate 1: Completeness score
        completeness = self._calculate_completeness(addr)
        if completeness >= self._min_completeness:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.COMPLETENESS_SCORE,
                    f"Completeness score: {completeness:.2f}",
                    score=completeness,
                    threshold=self._min_completeness,
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.COMPLETENESS_SCORE,
                    f"Completeness score {completeness:.2f} below threshold",
                    score=completeness,
                    threshold=self._min_completeness,
                    details={"missing": self._get_missing_components(addr)},
                )
            )

        # Gate 2: Confidence threshold
        # Use completeness as base confidence, adjusted for validation results
        confidence = self._calculate_confidence(addr, completeness)
        record.quality_score = confidence  # Update record

        if confidence >= self._min_confidence:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.CONFIDENCE_THRESHOLD,
                    f"Confidence: {confidence:.2f}",
                    score=confidence,
                    threshold=self._min_confidence,
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.CONFIDENCE_THRESHOLD,
                    f"Confidence {confidence:.2f} below threshold",
                    score=confidence,
                    threshold=self._min_confidence,
                )
            )

        # Gate 3: Puerto Rico urbanization
        is_pr = self._is_puerto_rico(addr)
        if is_pr:
            if addr.urbanization or not self._require_pr_urbanization:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.PR_URBANIZATION,
                        f"PR address with urbanization: {addr.urbanization}"
                        if addr.urbanization
                        else "PR urbanization not required",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.PR_URBANIZATION,
                        "Puerto Rico address missing urbanization (URB)",
                        details={"zip": addr.zip_code},
                    )
                )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.PR_URBANIZATION,
                    "Not a Puerto Rico address",
                )
            )

        # Gate 4: No invalid characters
        invalid_chars = self._find_invalid_chars(addr)
        if not invalid_chars:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.NO_INVALID_CHARS,
                    "No invalid characters found",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.NO_INVALID_CHARS,
                    f"Invalid characters found: {invalid_chars}",
                    details={"chars": invalid_chars},
                )
            )

        return self._create_result(gates_passed, gates_failed)

    def _calculate_completeness(self, addr: Any) -> float:
        """Calculate completeness score based on present components."""
        score = 0.0

        for component, weight in self.COMPONENT_WEIGHTS.items():
            value = getattr(addr, component, None)
            if value and str(value).strip():
                score += weight

        return min(score, 1.0)

    def _calculate_confidence(self, addr: Any, completeness: float) -> float:
        """Calculate overall confidence score.

        Combines completeness with quality signals.
        """
        confidence = completeness

        # Boost for having key fields
        if addr.zip_code and addr.street_name:
            confidence = min(confidence * 1.1, 1.0)

        # Penalty for very short fields (likely errors)
        if addr.city and len(addr.city) < 2:
            confidence *= 0.9
        if addr.street_name and len(addr.street_name) < 3:
            confidence *= 0.9

        # Boost for valid state
        if addr.state and len(addr.state) == 2:
            confidence = min(confidence * 1.05, 1.0)

        return round(confidence, 3)

    def _get_missing_components(self, addr: Any) -> list[str]:
        """Get list of missing/empty components."""
        missing = []
        for component in self.COMPONENT_WEIGHTS.keys():
            value = getattr(addr, component, None)
            if not value or not str(value).strip():
                missing.append(component)
        return missing

    def _is_puerto_rico(self, addr: Any) -> bool:
        """Check if address is in Puerto Rico."""
        # Check state
        if addr.state and addr.state.upper() == "PR":
            return True

        # Check ZIP code (PR ZIPs start with 006-009)
        if addr.zip_code:
            zip_prefix = addr.zip_code[:3]
            if zip_prefix in ("006", "007", "008", "009"):
                return True

        return False

    def _find_invalid_chars(self, addr: Any) -> list[str]:
        """Find invalid characters in address fields."""
        invalid = []

        # Characters that shouldn't appear in addresses
        invalid_pattern = r"[<>{}[\]|\\^~`@$%&*+=]"

        fields_to_check = [
            ("street_name", addr.street_name),
            ("city", addr.city),
            ("street_number", addr.street_number),
        ]

        for field_name, value in fields_to_check:
            if value:
                matches = re.findall(invalid_pattern, str(value))
                if matches:
                    invalid.extend(
                        f"{field_name}:{char}" for char in set(matches)
                    )

        return invalid
