"""Field extractors for various data formats.

Handles extraction of values from JSON, CSV, and XML structures
using field mappings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from icda.ingestion.schema.schema_models import (
    FieldMapping,
    SchemaMapping,
    CompositeFieldRule,
)

logger = logging.getLogger(__name__)


class FieldExtractor:
    """Extracts field values using schema mappings.

    Supports:
    - Simple field extraction
    - Nested JSON path extraction
    - Composite field parsing
    - Field transformations
    """

    def __init__(self, mapping: SchemaMapping):
        """Initialize with schema mapping.

        Args:
            mapping: SchemaMapping to use for extraction.
        """
        self._mapping = mapping
        self._field_map = {m.target_field: m for m in mapping.field_mappings}

    def extract(
        self,
        record: dict[str, Any],
        target_field: str,
    ) -> str | None:
        """Extract a single field value from record.

        Args:
            record: Source data record.
            target_field: Target field to extract.

        Returns:
            Extracted value or None.
        """
        field_mapping = self._field_map.get(target_field)
        if not field_mapping:
            return None

        # Get raw value
        if field_mapping.json_path:
            value = self._extract_nested(record, field_mapping.json_path)
        else:
            value = record.get(field_mapping.source_field)

        if value is None:
            return None

        # Apply transformation if specified
        if field_mapping.transform:
            value = self._apply_transform(str(value), field_mapping.transform)

        return str(value).strip() if value else None

    def extract_all(
        self,
        record: dict[str, Any],
    ) -> dict[str, str | None]:
        """Extract all mapped fields from record.

        Args:
            record: Source data record.

        Returns:
            Dictionary of target_field -> value.
        """
        result: dict[str, str | None] = {}

        for mapping in self._mapping.field_mappings:
            result[mapping.target_field] = self.extract(record, mapping.target_field)

        # Handle composite fields
        for rule in self._mapping.composite_rules:
            composite_value = record.get(rule.source_field)
            if composite_value:
                extracted = self._extract_composite(str(composite_value), rule)
                result.update(extracted)

        return result

    def _extract_nested(
        self,
        record: dict[str, Any],
        json_path: str,
    ) -> Any:
        """Extract value from nested JSON path.

        Args:
            record: Source record.
            json_path: Dot-separated path (e.g., "address.street").

        Returns:
            Extracted value or None.
        """
        parts = json_path.split(".")
        current = record

        for part in parts:
            if isinstance(current, dict):
                # Handle array index notation
                if "[" in part:
                    field_name, idx_str = part.rstrip("]").split("[")
                    current = current.get(field_name)
                    if isinstance(current, list):
                        try:
                            idx = int(idx_str)
                            current = current[idx] if idx < len(current) else None
                        except (ValueError, IndexError):
                            return None
                else:
                    current = current.get(part)
            else:
                return None

            if current is None:
                return None

        return current

    def _apply_transform(self, value: str, transform: str) -> str:
        """Apply transformation rule to value.

        Supported transforms:
        - upper: Convert to uppercase
        - lower: Convert to lowercase
        - strip: Strip whitespace
        - normalize_state: Normalize state to 2-letter code
        - normalize_zip: Normalize ZIP to 5-digit

        Args:
            value: Input value.
            transform: Transform rule name.

        Returns:
            Transformed value.
        """
        if transform == "upper":
            return value.upper()
        elif transform == "lower":
            return value.lower()
        elif transform == "strip":
            return value.strip()
        elif transform == "normalize_state":
            return self._normalize_state(value)
        elif transform == "normalize_zip":
            return self._normalize_zip(value)
        elif transform.startswith("regex:"):
            pattern = transform[6:]
            match = re.search(pattern, value)
            return match.group(1) if match else value
        else:
            return value

    def _normalize_state(self, value: str) -> str:
        """Normalize state to 2-letter abbreviation."""
        value = value.strip().upper()

        # Common state name mappings
        state_map = {
            "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
            "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT",
            "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
            "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
            "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME",
            "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI",
            "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO",
            "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
            "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
            "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND",
            "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA",
            "PUERTO RICO": "PR", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
            "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
            "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
            "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
        }

        if len(value) == 2:
            return value
        return state_map.get(value, value)

    def _normalize_zip(self, value: str) -> str:
        """Normalize ZIP code to 5 digits."""
        # Remove non-digits except hyphen
        cleaned = re.sub(r"[^\d-]", "", value)

        # Handle ZIP+4 format
        if "-" in cleaned:
            cleaned = cleaned.split("-")[0]

        # Take first 5 digits
        digits = re.sub(r"\D", "", cleaned)
        return digits[:5] if len(digits) >= 5 else digits

    def _extract_composite(
        self,
        value: str,
        rule: CompositeFieldRule,
    ) -> dict[str, str | None]:
        """Extract fields from composite value.

        Args:
            value: Composite address string.
            rule: Extraction rule.

        Returns:
            Dictionary of extracted fields.
        """
        result: dict[str, str | None] = {}

        if rule.extraction_pattern:
            # Use regex pattern
            match = re.match(rule.extraction_pattern, value)
            if match:
                groups = match.groups()
                for i, target in enumerate(rule.target_fields):
                    if i < len(groups):
                        result[target] = groups[i]
        else:
            # Use default US address parsing
            result = self._parse_us_address(value)

        return result

    def _parse_us_address(self, address: str) -> dict[str, str | None]:
        """Parse a US address string into components.

        Handles formats like:
        - "123 Main St, Anytown, NY 12345"
        - "123 Main Street Apt 4, City, ST 12345-6789"

        Args:
            address: Full address string.

        Returns:
            Dictionary of parsed components.
        """
        result: dict[str, str | None] = {
            "street_number": None,
            "street_name": None,
            "city": None,
            "state": None,
            "zip_code": None,
        }

        # Split by comma
        parts = [p.strip() for p in address.split(",")]

        if not parts:
            return result

        # Last part usually has state and ZIP
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            # Pattern: "ST 12345" or "State 12345-6789"
            state_zip_match = re.search(
                r"([A-Z]{2})\s*(\d{5}(?:-\d{4})?)",
                last_part,
                re.IGNORECASE,
            )
            if state_zip_match:
                result["state"] = state_zip_match.group(1).upper()
                result["zip_code"] = state_zip_match.group(2)[:5]

            # Second to last is usually city
            if len(parts) >= 3:
                result["city"] = parts[-2].strip()
            elif len(parts) == 2:
                # Try to extract city from last part before state
                city_match = re.match(r"^([^0-9]+)", last_part)
                if city_match:
                    potential_city = city_match.group(1).strip()
                    if len(potential_city) > 2:
                        result["city"] = potential_city

        # First part is street address
        street_part = parts[0].strip()

        # Extract street number
        number_match = re.match(r"^(\d+[A-Za-z]?)\s+(.+)", street_part)
        if number_match:
            result["street_number"] = number_match.group(1)
            result["street_name"] = number_match.group(2)
        else:
            result["street_name"] = street_part

        return result


def create_extractor(mapping: SchemaMapping) -> FieldExtractor:
    """Factory function for field extractor.

    Args:
        mapping: Schema mapping.

    Returns:
        Configured FieldExtractor.
    """
    return FieldExtractor(mapping)
