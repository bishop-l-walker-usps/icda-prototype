"""Ingestion enforcer pipeline.

5-stage quality validation for address data ingestion:
1. SchemaEnforcer - Validates mapped fields
2. NormalizationEnforcer - Parses and normalizes addresses
3. DuplicateEnforcer - Detects duplicates
4. QualityEnforcer - Scores address quality
5. ApprovalEnforcer - Final approval gate
"""

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    BaseIngestionEnforcer,
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)
from icda.ingestion.enforcers.ingestion_coordinator import IngestionEnforcerCoordinator

__all__ = [
    "BaseIngestionEnforcer",
    "IngestionGate",
    "IngestionGateResult",
    "IngestionEnforcerResult",
    "IngestionEnforcerCoordinator",
]
