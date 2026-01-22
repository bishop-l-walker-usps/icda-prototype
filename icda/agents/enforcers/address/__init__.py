"""Address Validation Enforcer Agents Package.

This package provides 6 specialized enforcer agents for comprehensive
address validation quality control.

Enforcers:
1. NormalizationEnforcerAgent - Validates address normalization (case, whitespace, etc.)
2. CompletionEnforcerAgent - Validates address completion (ZIP→city inference)
3. CorrectionEnforcerAgent - Validates address corrections (typos, misspellings)
4. MatchConfidenceEnforcerAgent - Validates match confidence and ambiguity
5. ValidationReportAgent - Generates comprehensive validation reports
6. BatchOrchestratorAgent - Coordinates all enforcers for batch processing

The AddressEnforcerCoordinator orchestrates all enforcers for single addresses.
"""

from .address_enforcer_coordinator import AddressEnforcerCoordinator
from .batch_orchestrator import BatchOrchestratorAgent, parse_addresses_from_file
from .completion_enforcer import CompletionEnforcerAgent
from .correction_enforcer import CorrectionEnforcerAgent
from .match_confidence_enforcer import MatchConfidenceEnforcerAgent
from .models import (
    AddressEnforcerGate,
    AddressFixSummary,
    AddressInput,
    BatchConfiguration,
    BatchProgress,
    BatchValidationReport,
    CompletionMetrics,
    ComponentQualityStats,
    ComponentType,
    CorrectionDetail,
    CorrectionMetrics,
    CorrectionType,
    ErrorTypeStats,
    MatchConfidenceMetrics,
    NormalizationMetrics,
)
from .normalization_enforcer import NormalizationEnforcerAgent
from .validation_report_agent import ValidationReportAgent

__all__ = [
    # Coordinator
    "AddressEnforcerCoordinator",
    # Enforcers
    "NormalizationEnforcerAgent",
    "CompletionEnforcerAgent",
    "CorrectionEnforcerAgent",
    "MatchConfidenceEnforcerAgent",
    "ValidationReportAgent",
    "BatchOrchestratorAgent",
    # Utilities
    "parse_addresses_from_file",
    # Models - Gates
    "AddressEnforcerGate",
    # Models - Types
    "CorrectionType",
    "ComponentType",
    # Models - Metrics
    "NormalizationMetrics",
    "CompletionMetrics",
    "CorrectionDetail",
    "CorrectionMetrics",
    "MatchConfidenceMetrics",
    # Models - Reports
    "AddressFixSummary",
    "ErrorTypeStats",
    "ComponentQualityStats",
    "BatchValidationReport",
    # Models - Batch
    "BatchProgress",
    "BatchConfiguration",
    "AddressInput",
]
