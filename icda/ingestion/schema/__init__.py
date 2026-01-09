"""AI-powered schema mapping for address data.

Automatically detects and maps address fields from unknown
data formats using LLM analysis.
"""

from icda.ingestion.schema.schema_models import (
    FieldMapping,
    SchemaMapping,
    MappingConfidence,
)
from icda.ingestion.schema.schema_mapper import AISchemaMapper

__all__ = [
    "FieldMapping",
    "SchemaMapping",
    "MappingConfidence",
    "AISchemaMapper",
]
