"""Schema enforcer - Stage 1.

Validates that schema mapping was applied and required fields are present.
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
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class SchemaEnforcer(BaseIngestionEnforcer):
    """Stage 1: Validates mapped fields and completeness.

    Gates:
    - FIELDS_MAPPED: Schema mapping was applied
    - REQUIRED_PRESENT: Required fields exist
    - TYPES_VALID: Field types are correct
    - SOURCE_ID_PRESENT: Record has source ID
    """

    __slots__ = ("_required_fields", "_optional_fields")

    def __init__(
        self,
        required_fields: list[str] | None = None,
        optional_fields: list[str] | None = None,
        enabled: bool = True,
    ):
        """Initialize schema enforcer.

        Args:
            required_fields: Fields that must be present.
            optional_fields: Fields that are optional.
            enabled: Whether enforcer is active.
        """
        super().__init__("schema_enforcer", enabled)
        self._required_fields = required_fields or ["street_name", "zip_code"]
        self._optional_fields = optional_fields or [
            "street_number", "city", "state", "unit"
        ]

    def get_gates(self) -> list[IngestionGate]:
        """Get gates this enforcer checks."""
        return [
            IngestionGate.FIELDS_MAPPED,
            IngestionGate.REQUIRED_PRESENT,
            IngestionGate.TYPES_VALID,
            IngestionGate.SOURCE_ID_PRESENT,
        ]

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Validate schema mapping.

        Args:
            record: IngestionRecord to validate.
            context: Additional context.

        Returns:
            IngestionEnforcerResult.
        """
        gates_passed: list[IngestionGateResult] = []
        gates_failed: list[IngestionGateResult] = []

        # Gate 1: Source ID present
        if record.source_id:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.SOURCE_ID_PRESENT,
                    f"Source ID present: {record.source_id}",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.SOURCE_ID_PRESENT,
                    "Missing source ID",
                )
            )

        # Gate 2: Fields mapped (check if parsed_address exists)
        if record.parsed_address:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.FIELDS_MAPPED,
                    "Schema mapping applied",
                )
            )

            # Gate 3: Required fields present
            missing_required = []
            for field in self._required_fields:
                value = getattr(record.parsed_address, field, None)
                if not value:
                    missing_required.append(field)

            if not missing_required:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.REQUIRED_PRESENT,
                        f"All required fields present: {self._required_fields}",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.REQUIRED_PRESENT,
                        f"Missing required fields: {missing_required}",
                        details={"missing": missing_required},
                    )
                )

            # Gate 4: Types valid (basic type checking)
            type_issues = self._validate_types(record)
            if not type_issues:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.TYPES_VALID,
                        "Field types are valid",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.TYPES_VALID,
                        f"Type validation issues: {type_issues}",
                        details={"issues": type_issues},
                    )
                )

        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.FIELDS_MAPPED,
                    "Schema mapping not applied - no parsed address",
                )
            )
            # Skip remaining gates if no mapping
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.REQUIRED_PRESENT,
                    "Cannot check required fields - no parsed address",
                )
            )
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.TYPES_VALID,
                    "Cannot validate types - no parsed address",
                )
            )

        return self._create_result(gates_passed, gates_failed)

    def _validate_types(self, record: IngestionRecord) -> list[str]:
        """Validate field types.

        Args:
            record: Record to validate.

        Returns:
            List of type validation issues.
        """
        issues: list[str] = []
        addr = record.parsed_address

        if not addr:
            return ["No parsed address"]

        # ZIP code should be numeric (with optional hyphen for ZIP+4)
        if addr.zip_code:
            import re
            if not re.match(r"^\d{5}(-\d{4})?$", addr.zip_code):
                # Check if at least has 5 digits
                if not re.match(r"^\d{5}", addr.zip_code):
                    issues.append(f"Invalid ZIP format: {addr.zip_code}")

        # State should be 2 letters
        if addr.state:
            if not (len(addr.state) == 2 and addr.state.isalpha()):
                issues.append(f"Invalid state format: {addr.state}")

        # Street number should start with a number
        if addr.street_number:
            if not addr.street_number[0].isdigit():
                issues.append(f"Street number doesn't start with digit: {addr.street_number}")

        return issues
