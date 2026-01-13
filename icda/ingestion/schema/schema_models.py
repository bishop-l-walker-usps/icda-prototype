"""Schema mapping data models.

Defines structures for field mappings, schema mappings,
and mapping confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MappingConfidence(str, Enum):
    """Confidence levels for schema mappings."""

    HIGH = "high"          # >= 0.9 - Very confident
    MEDIUM = "medium"      # >= 0.7 - Reasonably confident
    LOW = "low"            # >= 0.5 - Uncertain
    VERY_LOW = "very_low"  # < 0.5 - Guessing

    @classmethod
    def from_score(cls, score: float) -> MappingConfidence:
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return cls.HIGH
        elif score >= 0.7:
            return cls.MEDIUM
        elif score >= 0.5:
            return cls.LOW
        return cls.VERY_LOW


class AddressField(str, Enum):
    """Standard address field components."""

    STREET_NUMBER = "street_number"
    STREET_NAME = "street_name"
    STREET_TYPE = "street_type"
    UNIT = "unit"
    CITY = "city"
    STATE = "state"
    ZIP_CODE = "zip_code"
    ZIP_PLUS4 = "zip_plus4"
    COUNTRY = "country"
    URBANIZATION = "urbanization"
    FULL_ADDRESS = "full_address"  # Composite field


@dataclass(slots=True)
class FieldMapping:
    """Mapping from source field to address component.

    Attributes:
        source_field: Field name in source data.
        target_field: Target AddressField component.
        confidence: Mapping confidence score (0.0-1.0).
        transform: Optional transformation rule.
        json_path: JSON path for nested fields (e.g., "address.street").
        sample_values: Sample values seen during detection.
    """

    source_field: str
    target_field: str
    confidence: float = 1.0
    transform: str | None = None
    json_path: str | None = None
    sample_values: list[str] = field(default_factory=list)

    @property
    def confidence_level(self) -> MappingConfidence:
        """Get confidence level enum."""
        return MappingConfidence.from_score(self.confidence)

    @property
    def is_nested(self) -> bool:
        """Check if this is a nested JSON field."""
        return self.json_path is not None and "." in self.json_path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_field": self.source_field,
            "target_field": self.target_field,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "transform": self.transform,
            "json_path": self.json_path,
            "sample_values": self.sample_values[:3],  # Limit samples
        }


@dataclass(slots=True)
class CompositeFieldRule:
    """Rule for extracting from composite fields.

    Used when a single source field contains multiple address components
    (e.g., "123 Main St, Anytown, NY 12345").

    Attributes:
        source_field: Field containing composite address.
        extraction_pattern: Regex or rule for extraction.
        target_fields: Fields to extract to.
    """

    source_field: str
    extraction_pattern: str | None = None
    target_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_field": self.source_field,
            "extraction_pattern": self.extraction_pattern,
            "target_fields": self.target_fields,
        }


@dataclass(slots=True)
class SchemaMapping:
    """Complete schema mapping for a data source.

    Represents the learned mapping from a source's schema
    to standard address fields.

    Attributes:
        source_name: Identifier for the data source.
        source_format: Format type (json, csv, xml).
        field_mappings: List of field-to-field mappings.
        composite_rules: Rules for composite field extraction.
        unmapped_fields: Source fields that weren't mapped.
        confidence: Overall mapping confidence (0.0-1.0).
        detected_at: When mapping was detected.
        last_used: When mapping was last used.
        use_count: Number of times mapping has been used.
        sample_record: Sample record used for detection.
    """

    source_name: str
    source_format: str
    field_mappings: list[FieldMapping] = field(default_factory=list)
    composite_rules: list[CompositeFieldRule] = field(default_factory=list)
    unmapped_fields: list[str] = field(default_factory=list)
    confidence: float = 0.0
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used: str | None = None
    use_count: int = 0
    sample_record: dict[str, Any] | None = None

    @property
    def confidence_level(self) -> MappingConfidence:
        """Get confidence level enum."""
        return MappingConfidence.from_score(self.confidence)

    @property
    def is_complete(self) -> bool:
        """Check if mapping has minimum required fields."""
        required = {"street_name", "zip_code"}
        mapped_targets = {m.target_field for m in self.field_mappings}
        return required.issubset(mapped_targets)

    @property
    def mapped_fields(self) -> list[str]:
        """Get list of mapped target fields."""
        return [m.target_field for m in self.field_mappings]

    def get_mapping(self, target_field: str) -> FieldMapping | None:
        """Get mapping for a target field.

        Args:
            target_field: Target AddressField name.

        Returns:
            FieldMapping or None if not mapped.
        """
        for mapping in self.field_mappings:
            if mapping.target_field == target_field:
                return mapping
        return None

    def record_use(self) -> None:
        """Record that this mapping was used."""
        self.use_count += 1
        self.last_used = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "source_name": self.source_name,
            "source_format": self.source_format,
            "field_mappings": [m.to_dict() for m in self.field_mappings],
            "composite_rules": [r.to_dict() for r in self.composite_rules],
            "unmapped_fields": self.unmapped_fields,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "is_complete": self.is_complete,
            "detected_at": self.detected_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaMapping:
        """Create from dictionary."""
        field_mappings = [
            FieldMapping(
                source_field=m["source_field"],
                target_field=m["target_field"],
                confidence=m.get("confidence", 1.0),
                transform=m.get("transform"),
                json_path=m.get("json_path"),
                sample_values=m.get("sample_values", []),
            )
            for m in data.get("field_mappings", [])
        ]

        composite_rules = [
            CompositeFieldRule(
                source_field=r["source_field"],
                extraction_pattern=r.get("extraction_pattern"),
                target_fields=r.get("target_fields", []),
            )
            for r in data.get("composite_rules", [])
        ]

        return cls(
            source_name=data["source_name"],
            source_format=data.get("source_format", "json"),
            field_mappings=field_mappings,
            composite_rules=composite_rules,
            unmapped_fields=data.get("unmapped_fields", []),
            confidence=data.get("confidence", 0.0),
            detected_at=data.get("detected_at", datetime.utcnow().isoformat()),
            last_used=data.get("last_used"),
            use_count=data.get("use_count", 0),
        )
