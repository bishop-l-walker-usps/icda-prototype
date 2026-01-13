"""Normalization enforcer - Stage 2.

Uses AddressNormalizer to parse and validate addresses.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    BaseIngestionEnforcer,
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)

if TYPE_CHECKING:
    from icda.address_normalizer import AddressNormalizer
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class NormalizationEnforcer(BaseIngestionEnforcer):
    """Stage 2: Uses AddressNormalizer, validates parsing.

    Gates:
    - ADDRESS_PARSEABLE: Raw address can be parsed
    - COMPONENTS_EXTRACTED: Key components were found
    - STATE_NORMALIZED: State is valid 2-letter code
    - ZIP_FORMAT_VALID: ZIP is 5 or 9 digits
    - STREET_NAME_PRESENT: Street name was extracted
    """

    __slots__ = ("_normalizer", "_require_state", "_require_zip")

    def __init__(
        self,
        normalizer: AddressNormalizer | None = None,
        require_state: bool = True,
        require_zip: bool = True,
        enabled: bool = True,
    ):
        """Initialize normalization enforcer.

        Args:
            normalizer: AddressNormalizer instance.
            require_state: Whether state is required.
            require_zip: Whether ZIP code is required.
            enabled: Whether enforcer is active.
        """
        super().__init__("normalization_enforcer", enabled)
        self._normalizer = normalizer
        self._require_state = require_state
        self._require_zip = require_zip

    def get_gates(self) -> list[IngestionGate]:
        """Get gates this enforcer checks."""
        return [
            IngestionGate.ADDRESS_PARSEABLE,
            IngestionGate.COMPONENTS_EXTRACTED,
            IngestionGate.STATE_NORMALIZED,
            IngestionGate.ZIP_FORMAT_VALID,
            IngestionGate.STREET_NAME_PRESENT,
        ]

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Validate address normalization.

        Args:
            record: IngestionRecord to validate.
            context: Additional context.

        Returns:
            IngestionEnforcerResult.
        """
        gates_passed: list[IngestionGateResult] = []
        gates_failed: list[IngestionGateResult] = []
        modified_record = None

        # If no parsed address, try to normalize from raw
        if not record.parsed_address and record.source_record.raw_address:
            if self._normalizer:
                try:
                    parsed = self._normalizer.normalize(
                        record.source_record.raw_address
                    )
                    record.parsed_address = parsed
                    modified_record = record
                except Exception as e:
                    logger.warning(f"Normalization failed: {e}")

        addr = record.parsed_address

        # Gate 1: Address parseable
        if addr:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.ADDRESS_PARSEABLE,
                    "Address parsed successfully",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.ADDRESS_PARSEABLE,
                    "Could not parse address",
                )
            )
            # Return early - can't check other gates
            return self._create_result(gates_passed, gates_failed, modified_record)

        # Gate 2: Components extracted
        components_found = []
        if addr.street_number:
            components_found.append("street_number")
        if addr.street_name:
            components_found.append("street_name")
        if addr.city:
            components_found.append("city")
        if addr.state:
            components_found.append("state")
        if addr.zip_code:
            components_found.append("zip_code")

        min_components = 2  # At least 2 components
        if len(components_found) >= min_components:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.COMPONENTS_EXTRACTED,
                    f"Extracted {len(components_found)} components: {components_found}",
                    score=len(components_found) / 5,  # Max 5 core components
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.COMPONENTS_EXTRACTED,
                    f"Only {len(components_found)} components extracted",
                    score=len(components_found) / 5,
                    threshold=min_components / 5,
                    details={"found": components_found},
                )
            )

        # Gate 3: State normalized
        if addr.state:
            if self._is_valid_state(addr.state):
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.STATE_NORMALIZED,
                        f"Valid state code: {addr.state}",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.STATE_NORMALIZED,
                        f"Invalid state code: {addr.state}",
                    )
                )
        elif self._require_state:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.STATE_NORMALIZED,
                    "State is required but missing",
                )
            )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.STATE_NORMALIZED,
                    "State not required and not present",
                )
            )

        # Gate 4: ZIP format valid
        if addr.zip_code:
            if self._is_valid_zip(addr.zip_code):
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.ZIP_FORMAT_VALID,
                        f"Valid ZIP code: {addr.zip_code}",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.ZIP_FORMAT_VALID,
                        f"Invalid ZIP format: {addr.zip_code}",
                    )
                )
        elif self._require_zip:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.ZIP_FORMAT_VALID,
                    "ZIP code is required but missing",
                )
            )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.ZIP_FORMAT_VALID,
                    "ZIP not required and not present",
                )
            )

        # Gate 5: Street name present
        if addr.street_name:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.STREET_NAME_PRESENT,
                    f"Street name present: {addr.street_name}",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.STREET_NAME_PRESENT,
                    "Street name is required but missing",
                )
            )

        return self._create_result(gates_passed, gates_failed, modified_record)

    def _is_valid_state(self, state: str) -> bool:
        """Check if state is a valid 2-letter code."""
        valid_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI",
            "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
            "WY", "DC", "GU", "VI", "AS", "MP",
        }
        return state.upper() in valid_states

    def _is_valid_zip(self, zip_code: str) -> bool:
        """Check if ZIP code is valid format."""
        import re
        # 5 digits or 5+4 format
        return bool(re.match(r"^\d{5}(-\d{4})?$", zip_code))
