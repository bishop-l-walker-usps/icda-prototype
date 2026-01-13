"""AI-powered schema mapper.

Uses LLM to detect and map address fields from unknown data formats.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from icda.address_models import ParsedAddress
from icda.ingestion.schema.schema_models import (
    FieldMapping,
    SchemaMapping,
    CompositeFieldRule,
)
from icda.ingestion.schema.mapping_cache import MappingCache
from icda.ingestion.schema.field_extractors import FieldExtractor

if TYPE_CHECKING:
    from icda.nova import NovaClient

logger = logging.getLogger(__name__)


# Common field patterns for heuristic detection
FIELD_PATTERNS = {
    "street_number": [
        r"^(house|building|street|addr)[\s_-]*(num|number|no|#)$",
        r"^(num|number|no)$",
    ],
    "street_name": [
        r"^(street|str|road|rd|address)[\s_-]*(name)?$",
        r"^(addr|address)[\s_-]*(line)?[\s_-]*(1|one)?$",
    ],
    "unit": [
        r"^(unit|apt|apartment|suite|ste|floor|fl|room|rm)[\s_-]*(num|number|no|#)?$",
        r"^(addr|address)[\s_-]*(line)?[\s_-]*(2|two)$",
    ],
    "city": [
        r"^(city|town|municipality|locality)$",
    ],
    "state": [
        r"^(state|province|region|st)$",
    ],
    "zip_code": [
        r"^(zip|zipcode|zip[\s_-]*code|postal[\s_-]*code|postcode)$",
    ],
    "zip_plus4": [
        r"^(zip[\s_-]*(plus)?[\s_-]*4|zip4|plus[\s_-]*4)$",
    ],
    "country": [
        r"^(country|nation|country[\s_-]*code)$",
    ],
    "urbanization": [
        r"^(urbanization|urb|urbanizacion)$",
    ],
    "full_address": [
        r"^(full[\s_-]*address|complete[\s_-]*address|address[\s_-]*full)$",
        r"^(mailing[\s_-]*address|delivery[\s_-]*address)$",
    ],
}


class AISchemaMapper:
    """Uses LLM to detect and map address fields from unknown schemas.

    Features:
    - Analyzes sample records to detect patterns
    - Uses Nova LLM for uncertain mappings
    - Caches learned mappings for reuse
    - Supports JSON, CSV, and nested structures
    - Falls back to heuristic matching when LLM unavailable
    """

    __slots__ = ("_nova", "_cache", "_min_confidence", "_use_llm")

    def __init__(
        self,
        nova_client: NovaClient | None = None,
        cache: MappingCache | None = None,
        min_confidence: float = 0.7,
        use_llm: bool = True,
    ):
        """Initialize schema mapper.

        Args:
            nova_client: NovaClient for LLM analysis.
            cache: MappingCache for persistence.
            min_confidence: Minimum confidence to cache mapping.
            use_llm: Whether to use LLM for detection.
        """
        self._nova = nova_client
        self._cache = cache or MappingCache()
        self._min_confidence = min_confidence
        self._use_llm = use_llm and nova_client is not None

    async def detect_mapping(
        self,
        source_name: str,
        sample_records: list[dict[str, Any]],
        source_format: str = "json",
    ) -> SchemaMapping:
        """Detect field mapping from sample records.

        1. Check cache for existing mapping
        2. If not cached, use heuristics + LLM to detect
        3. Validate mapping against samples
        4. Cache if confidence is sufficient

        Args:
            source_name: Identifier for data source.
            sample_records: Sample records for analysis.
            source_format: Format (json, csv).

        Returns:
            SchemaMapping for the source.
        """
        # Check cache
        cached = await self._cache.get(source_name)
        if cached and cached.confidence >= self._min_confidence:
            cached.record_use()
            logger.info(f"Using cached mapping for {source_name}")
            return cached

        if not sample_records:
            logger.warning(f"No sample records for {source_name}")
            return SchemaMapping(source_name=source_name, source_format=source_format)

        # Extract field names from samples
        field_names = self._extract_field_names(sample_records)

        # Try heuristic mapping first
        mappings = self._heuristic_mapping(field_names, sample_records)

        # Use LLM for uncertain fields
        if self._use_llm and self._nova:
            mappings = await self._llm_enhance_mapping(
                mappings, field_names, sample_records
            )

        # Check for composite address fields
        composite_rules = self._detect_composite_fields(field_names, sample_records)

        # Calculate overall confidence
        confidence = self._calculate_confidence(mappings)

        # Build schema mapping
        schema_mapping = SchemaMapping(
            source_name=source_name,
            source_format=source_format,
            field_mappings=mappings,
            composite_rules=composite_rules,
            unmapped_fields=self._find_unmapped(field_names, mappings),
            confidence=confidence,
            sample_record=sample_records[0] if sample_records else None,
        )

        # Cache if confidence sufficient
        if confidence >= self._min_confidence:
            await self._cache.put(source_name, schema_mapping)
            logger.info(
                f"Cached mapping for {source_name} (confidence={confidence:.2f})"
            )

        return schema_mapping

    def apply_mapping(
        self,
        record: dict[str, Any],
        mapping: SchemaMapping,
    ) -> ParsedAddress:
        """Apply schema mapping to extract ParsedAddress.

        Args:
            record: Source data record.
            mapping: Schema mapping to apply.

        Returns:
            ParsedAddress with extracted components.
        """
        extractor = FieldExtractor(mapping)
        extracted = extractor.extract_all(record)

        # Build ParsedAddress
        return ParsedAddress(
            raw=json.dumps(record) if isinstance(record, dict) else str(record),
            street_number=extracted.get("street_number"),
            street_name=extracted.get("street_name"),
            street_type=extracted.get("street_type"),
            unit=extracted.get("unit"),
            city=extracted.get("city"),
            state=extracted.get("state"),
            zip_code=extracted.get("zip_code"),
            zip_plus4=extracted.get("zip_plus4"),
            country=extracted.get("country", "US"),
            urbanization=extracted.get("urbanization"),
        )

    def _extract_field_names(
        self,
        records: list[dict[str, Any]],
    ) -> set[str]:
        """Extract all field names from sample records."""
        fields: set[str] = set()

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_name = f"{prefix}.{key}" if prefix else key
                    fields.add(field_name)
                    extract_recursive(value, field_name)
            elif isinstance(obj, list) and obj:
                extract_recursive(obj[0], prefix)

        for record in records:
            extract_recursive(record)

        return fields

    def _heuristic_mapping(
        self,
        field_names: set[str],
        sample_records: list[dict[str, Any]],
    ) -> list[FieldMapping]:
        """Apply heuristic pattern matching for field mapping."""
        mappings: list[FieldMapping] = []
        mapped_targets: set[str] = set()

        for field_name in field_names:
            # Normalize field name for matching
            normalized = field_name.lower().replace("-", "_").replace(" ", "_")
            base_name = normalized.split(".")[-1]  # Get leaf name for nested

            for target_field, patterns in FIELD_PATTERNS.items():
                if target_field in mapped_targets:
                    continue

                for pattern in patterns:
                    if re.match(pattern, base_name, re.IGNORECASE):
                        # Get sample values
                        samples = self._get_sample_values(
                            field_name, sample_records, limit=3
                        )

                        mappings.append(
                            FieldMapping(
                                source_field=field_name,
                                target_field=target_field,
                                confidence=0.85,  # Heuristic confidence
                                json_path=field_name if "." in field_name else None,
                                sample_values=samples,
                            )
                        )
                        mapped_targets.add(target_field)
                        break

        return mappings

    async def _llm_enhance_mapping(
        self,
        existing_mappings: list[FieldMapping],
        field_names: set[str],
        sample_records: list[dict[str, Any]],
    ) -> list[FieldMapping]:
        """Use LLM to enhance or fill in uncertain mappings."""
        if not self._nova:
            return existing_mappings

        # Find unmapped fields
        mapped_sources = {m.source_field for m in existing_mappings}
        unmapped = field_names - mapped_sources

        if not unmapped:
            return existing_mappings

        # Build prompt for LLM
        prompt = self._build_llm_prompt(unmapped, sample_records)

        try:
            # Call Nova for analysis
            response = await self._nova.analyze_text(prompt)

            if response:
                new_mappings = self._parse_llm_response(response, unmapped, sample_records)
                existing_mappings.extend(new_mappings)

        except Exception as e:
            logger.warning(f"LLM schema detection failed: {e}")

        return existing_mappings

    def _build_llm_prompt(
        self,
        unmapped_fields: set[str],
        sample_records: list[dict[str, Any]],
    ) -> str:
        """Build prompt for LLM field analysis."""
        # Get sample values for unmapped fields
        field_samples = {}
        for field in unmapped_fields:
            samples = self._get_sample_values(field, sample_records, limit=3)
            if samples:
                field_samples[field] = samples

        prompt = """Analyze these data fields and determine which standard US address components they map to.

Standard address components:
- street_number: House/building number (e.g., "123", "456A")
- street_name: Street name (e.g., "Main Street", "Oak Ave")
- street_type: Street type suffix (e.g., "St", "Ave", "Blvd")
- unit: Apartment/suite number (e.g., "Apt 4", "Suite 100")
- city: City name
- state: State abbreviation (2 letters)
- zip_code: 5-digit ZIP code
- zip_plus4: 4-digit ZIP extension
- urbanization: Puerto Rico urbanization (URB)
- full_address: Complete address in single field
- none: Not an address field

Fields to analyze:
"""
        for field, samples in field_samples.items():
            prompt += f"\n{field}: {samples}"

        prompt += """

Respond in JSON format:
{
  "mappings": [
    {"source": "field_name", "target": "component_name", "confidence": 0.9}
  ]
}

Only include fields that are address-related. Use "none" for non-address fields."""

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        unmapped_fields: set[str],
        sample_records: list[dict[str, Any]],
    ) -> list[FieldMapping]:
        """Parse LLM response into FieldMappings."""
        mappings: list[FieldMapping] = []

        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return mappings

            data = json.loads(json_match.group())

            for item in data.get("mappings", []):
                source = item.get("source")
                target = item.get("target")
                confidence = item.get("confidence", 0.7)

                if source in unmapped_fields and target and target != "none":
                    samples = self._get_sample_values(source, sample_records, limit=3)
                    mappings.append(
                        FieldMapping(
                            source_field=source,
                            target_field=target,
                            confidence=confidence,
                            json_path=source if "." in source else None,
                            sample_values=samples,
                        )
                    )

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")

        return mappings

    def _detect_composite_fields(
        self,
        field_names: set[str],
        sample_records: list[dict[str, Any]],
    ) -> list[CompositeFieldRule]:
        """Detect fields that contain complete addresses."""
        rules: list[CompositeFieldRule] = []

        for field_name in field_names:
            samples = self._get_sample_values(field_name, sample_records, limit=3)

            for sample in samples:
                if self._looks_like_full_address(sample):
                    rules.append(
                        CompositeFieldRule(
                            source_field=field_name,
                            target_fields=[
                                "street_number",
                                "street_name",
                                "city",
                                "state",
                                "zip_code",
                            ],
                        )
                    )
                    break

        return rules

    def _looks_like_full_address(self, value: str) -> bool:
        """Check if value looks like a complete address."""
        if not value or len(value) < 15:
            return False

        # Check for common address patterns
        patterns = [
            r"\d+\s+\w+.*,\s*\w+,?\s*[A-Z]{2}\s*\d{5}",  # 123 Main St, City, ST 12345
            r"\d+\s+\w+.*\s+[A-Z]{2}\s+\d{5}",  # 123 Main St City ST 12345
        ]

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def _get_sample_values(
        self,
        field_name: str,
        records: list[dict[str, Any]],
        limit: int = 3,
    ) -> list[str]:
        """Extract sample values for a field."""
        samples: list[str] = []

        for record in records[:limit * 2]:
            value = self._get_nested_value(record, field_name)
            if value and str(value).strip():
                samples.append(str(value).strip())
                if len(samples) >= limit:
                    break

        return samples

    def _get_nested_value(
        self,
        record: dict[str, Any],
        field_name: str,
    ) -> Any:
        """Get value from potentially nested field."""
        parts = field_name.split(".")
        current = record

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _calculate_confidence(self, mappings: list[FieldMapping]) -> float:
        """Calculate overall mapping confidence."""
        if not mappings:
            return 0.0

        # Required fields
        required = {"street_name", "zip_code"}
        mapped_targets = {m.target_field for m in mappings}

        has_required = required.issubset(mapped_targets)

        # Average confidence of mappings
        avg_conf = sum(m.confidence for m in mappings) / len(mappings)

        # Boost if required fields present
        if has_required:
            return min(avg_conf * 1.1, 1.0)

        return avg_conf * 0.8

    def _find_unmapped(
        self,
        field_names: set[str],
        mappings: list[FieldMapping],
    ) -> list[str]:
        """Find source fields that weren't mapped."""
        mapped_sources = {m.source_field for m in mappings}
        return [f for f in field_names if f not in mapped_sources]
